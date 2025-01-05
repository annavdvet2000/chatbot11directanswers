const express = require('express');
const cors = require('cors');
const corsOptions = {
    origin: 'https://chatbot11directanswers.netlify.app',
    methods: ['GET', 'POST'],
    credentials: true,
    allowedHeaders: ['Content-Type', 'Authorization'],
    preflightContinue: false,
    optionsSuccessStatus: 204
};
const dotenv = require('dotenv');
const OpenAI = require('openai');
const path = require('path');
const fs = require('fs');
const csv = require('csv-parse/sync');
const { Pool } = require('pg'); // Add this line

// Load environment variables
dotenv.config();

// Add PostgreSQL configuration
const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: {
        rejectUnauthorized: false
    }
});

// Define chatbot ID
const CHATBOT_ID = 'direct-answers-bot';

// Create database table
pool.query(`
    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        qualtrics_id VARCHAR(255),
        session_id VARCHAR(255),
        role VARCHAR(10),
        content TEXT,
        chatbot_id VARCHAR(50),
        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
`).catch(console.error);

// Rest of your AISearchEngine class stays exactly the same
class AISearchEngine {
    // ... [Your existing AISearchEngine class code stays the same]
}

// Your existing ensureCompleteResponse function stays the same
function ensureCompleteResponse(text) {
    // ... [Your existing ensureCompleteResponse function code stays the same]
}

const app = express();
app.use(cors(corsOptions));
app.use(express.json());

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

const searchEngine = new AISearchEngine(openai);
searchEngine.initialize().catch(console.error);

const sessions = new Map();

app.get('/', (req, res) => {
    res.json({ message: 'API is running' });
});

app.get('/api/chat', (req, res) => {
    res.json({ message: 'Please use POST method for chat requests' });
});

// Modified chat endpoint
app.post('/api/chat', async (req, res) => {
    try {
        const { question, sessionId } = req.body;
        const qualtricsId = req.body.qualtricsId || 'unknown';
        
        if (!question) {
            return res.status(400).json({ error: 'Question is required' });
        }

        // Store user's question in database
        await pool.query(
            'INSERT INTO chat_messages (qualtrics_id, session_id, role, content, chatbot_id) VALUES ($1, $2, $3, $4, $5)',
            [qualtricsId, sessionId, 'user', question, CHATBOT_ID]
        );

        let sessionHistory = sessions.get(sessionId) || [];
        const relevantContext = await searchEngine.findRelevantContext(question, sessionHistory);

        const completion = await openai.chat.completions.create({
            model: "gpt-4-turbo-preview",
            messages: [
                {
                    role: "system",
                    content: `You are a helpful assistant analyzing oral history interviews. Respond warmly to greetings or friendly messages (e.g., "hi," "hello," "how are you?"). Follow these rules strictly: 

CRITICAL RULES:
- ALWAYS start your response by citing the specific interview(s) you're drawing from
- Give ONE clear, definitive answer in the first sentence
- Use this format for citations: "From Interview #[X] with [Name]:"
- For multiple sources: "From Interview #[X] with [Name], and Interview #[Z] with [Name]:"
- After the citation, provide your concise answer
- Never make claims without citing specific interviews 
- If you can't find relevant information, say "I don't find information about this in the interviews"
- For comparative questions, cite both interviews before making any comparison
- If asked 'why', always point back to specific interviews and pages

WHICH/WHO QUESTIONS BETWEEN PEOPLE:
- ALWAYS choose one person as the primary figure based on:
   - Frequency of mention in relevant context
   - Scope and scale of their involvement
   - Whether it was their main focus vs. one of many activities
   - Direct vs. indirect involvement

HANDLING FOLLOW-UP QUESTIONS:
- Review previous exchanges to understand the context
- For "why" questions, refer back to the specific evidence from previously cited interviews
- If a follow-up question is unclear, ask for clarification about which aspect they want to know more about
- Always maintain continuity with previous responses
- If the follow-up requires new information not covered in previous responses, search for and cite new relevant passages

Example good response:
"From Interview #4 with Jean Carlomusto, page 12: She primarily worked on AIDS education videos at GMHC."

Example bad response:
"Jean Carlomusto worked on AIDS education videos at GMHC." (missing citation)

Only use information from the provided context. Here is the relevant context:\n\n${relevantContext}`
                },
                ...sessionHistory,
                {
                    role: "user",
                    content: question
                }
            ],
            temperature: 0.7,
            max_tokens: 150,
            presence_penalty: 1.0,
            frequency_penalty: 1.0
        });

        let response = completion.choices[0].message.content;
        response = ensureCompleteResponse(response);

        // Store bot's response in database
        await pool.query(
            'INSERT INTO chat_messages (qualtrics_id, session_id, role, content, chatbot_id) VALUES ($1, $2, $3, $4, $5)',
            [qualtricsId, sessionId, 'assistant', response, CHATBOT_ID]
        );

        // Store complete history without truncation
        sessionHistory = [
            ...sessionHistory,
            { role: "user", content: question },
            { role: "assistant", content: response }
        ];
        sessions.set(sessionId, sessionHistory);

        res.json({ response });

    } catch (error) {
        console.error('Error in chat endpoint:', error);
        
        if (error.response) {
            res.status(error.response.status).json({
                error: error.response.data.error.message,
                status: 'error'
            });
        } else {
            res.status(500).json({
                error: 'An error occurred while processing your request',
                status: 'error'
            });
        }
    }
});

// Add new endpoint to get chat history
app.get('/api/chat/history/:qualtricsId', async (req, res) => {
    try {
        const result = await pool.query(
            'SELECT * FROM chat_messages WHERE qualtrics_id = $1 ORDER BY timestamp',
            [req.params.qualtricsId]
        );
        
        res.json(result.rows);
    } catch (error) {
        console.error('Error getting chat history:', error);
        res.status(500).json({ error: 'Failed to get chat history' });
    }
});

app.use((req, res) => {
    res.status(404).json({ error: 'Route not found' });
});

app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ 
        error: 'Something broke!',
        details: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on ${PORT}`);
    console.log('Press Ctrl+C to stop the server');
});

process.on('unhandledRejection', (error) => {
    console.error('Unhandled Promise Rejection:', error);
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});