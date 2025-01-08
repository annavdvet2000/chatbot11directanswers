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
const { Pool } = require('pg');

// Load environment variables
dotenv.config();

// PostgreSQL configuration
const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: {
        rejectUnauthorized: false
    }
});

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

// Define chatbot ID
const CHATBOT_ID = 'direct-answers-bot';

class AISearchEngine {
    constructor(openai) {
        this.openai = openai;
        this.embeddings = [];
        this.texts = [];
        this.metadata = new Map();
        this.chunkMetadata = [];
    }

    async initialize() {
        try {
            const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'embeddings.json'), 'utf8'));
            this.embeddings = data.embeddings;
            this.texts = data.texts;
            this.chunkMetadata = data.metadata;

            console.log("First few texts entries:");
            this.texts.slice(0, 3).forEach((text, i) => {
                console.log(`Text ${i + 1} starts with: ${text.substring(0, 100)}...`);
            });

            const metadataFile = fs.readFileSync(path.join(__dirname, 'metadata.csv'), 'utf8');
            const records = csv.parse(metadataFile, {
                columns: true,
                skip_empty_lines: true
            });
            
            console.log("\nFirst few metadata records:");
            records.slice(0, 3).forEach((record, i) => {
                console.log(`Record ${i + 1}: ${record.name}`);
            });

            records.forEach((record, index) => {
                const documentId = (index + 1).toString();
                this.metadata.set(documentId, record);
            });
            
            console.log(`Loaded ${this.embeddings.length} embeddings and ${this.metadata.size} metadata records`);
        } catch (error) {
            console.error('Failed to load data:', error);
            throw error;
        }
    }

    findAllNamesInQuestion(question) {
        const normalizedQuestion = question.toLowerCase();
        const matches = [];
        
        for (const [id, record] of this.metadata.entries()) {
            const normalizedName = record.name.toLowerCase();
            
            if (normalizedQuestion.includes(normalizedName)) {
                matches.push({ id, record });
                continue;
            }
            
            const nameParts = normalizedName.split(' ');
            for (const part of nameParts) {
                if (part.length > 2 && normalizedQuestion.includes(part.toLowerCase())) {
                    matches.push({ id, record });
                    break;
                }
            }
        }
        return matches;
    }

    async findRelevantContext(question, history = []) {
        try {
            const recentHistory = history.slice(-4);
            let contextualQuery = question;

            if (question.length < 60 || 
                question.toLowerCase().startsWith('why') || 
                question.toLowerCase().startsWith('how') || 
                !question.includes('?')) {
                
                const previousExchanges = recentHistory
                    .map(msg => msg.content)
                    .join(' ');
                contextualQuery = `${previousExchanges} ${question}`;
            }

            const isComparative = contextualQuery.toLowerCase().includes('between') || 
                                contextualQuery.toLowerCase().includes('compare');
            
            const names = this.findAllNamesInQuestion(contextualQuery);
            
            if ((isComparative || names.length > 1) && names.length >= 2) {
                const contexts = [];
                for (const match of names) {
                    const documentName = `document${match.id}.pdf`;
                    const questionEmbedding = await this.getEmbedding(contextualQuery);
                    const similarContent = await this.findSimilarContent(questionEmbedding, documentName);
                    
                    if (similarContent.length > 0) {
                        contexts.push({
                            interviewId: match.id,
                            interviewee: match.record.name,
                            date: match.record.date,
                            content: similarContent.map(item => ({
                                text: item.text,
                                page: item.metadata.page,
                                score: item.score
                            }))
                        });
                    }
                }
                
                return contexts.map(context => 
                    `Interview ${context.interviewId} with ${context.interviewee} (${context.date}):\n${
                        context.content.map(c => `[Page ${c.page}] ${c.text}`).join('\n\n')
                    }`
                ).join('\n\n---\n\n');
            }
            
            const nameMatch = names[0];
            if (nameMatch) {
                const { id, record } = nameMatch;
                console.log(`Found match for person: ${record.name} (ID: ${id})`);
                
                const documentName = `document${id}.pdf`;
                const questionEmbedding = await this.getEmbedding(contextualQuery);
                const similarContent = await this.findSimilarContent(questionEmbedding, documentName);
                
                if (similarContent.length > 0) {
                    return similarContent.map(item => 
                        `Interview ${id} with ${record.name} (${record.date}):\n[Page ${item.metadata.page}] ${item.text}`
                    ).join('\n\n');
                }
            }

            const questionEmbedding = await this.getEmbedding(contextualQuery);
            const similarContent = await this.findSimilarContent(questionEmbedding, null);
            
            const groupedResults = {};
            similarContent.forEach(item => {
                const source = item.metadata.source;
                if (!groupedResults[source]) {
                    groupedResults[source] = [];
                }
                groupedResults[source].push(item);
            });

            if (Object.keys(groupedResults).length > 0) {
                const bestSource = Object.entries(groupedResults)
                    .sort((a, b) => b[1].length - a[1].length)[0];
                
                const documentId = bestSource[0].match(/document(\d+)\.pdf/)[1];
                const record = this.metadata.get(documentId);
                
                const relevantChunks = bestSource[1];
                return relevantChunks.map(item => 
                    `Interview ${documentId} with ${record.name} (${record.date}):\n[Page ${item.metadata.page}] ${item.text}`
                ).join('\n\n');
            }

            return null;
        } catch (error) {
            console.error('Error finding relevant context:', error);
            throw error;
        }
    }

    async findSimilarContent(queryEmbedding, sourceName = null) {
        const similarities = this.embeddings.map((emb, idx) => ({
            score: this.cosineSimilarity(queryEmbedding, emb),
            text: this.texts[idx],
            metadata: this.chunkMetadata[idx]
        }));

        let results = similarities;
        if (sourceName) {
            results = similarities.filter(item => 
                item.metadata && item.metadata.source === sourceName
            );
        }

        return results
            .sort((a, b) => b.score - a.score)
            .slice(0, 5);
    }

    cosineSimilarity(vecA, vecB) {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (normA * normB);
    }

    async getEmbedding(text) {
        const response = await this.openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text,
        });
        return response.data[0].embedding;
    }
}

function ensureCompleteResponse(text) {
    text = text.replace(/\.{3,}$/, '');
    const sentenceEndings = ['.', '!', '?'];
    
    if (sentenceEndings.some(ending => text.endsWith(ending))) {
        return text;
    }

    let lastCompleteIndex = -1;
    for (const ending of sentenceEndings) {
        const index = text.lastIndexOf(ending);
        if (index > lastCompleteIndex) {
            lastCompleteIndex = index;
        }
    }

    if (lastCompleteIndex !== -1) {
        text = text.substring(0, lastCompleteIndex + 1);
    } else {
        if (text.trim().length > 0) {
            text = text.trim() + '.';
        }
    }

    return text;
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

RESPONSE STRUCTURE:
1. Start with citation and clear answer
2. Provide brief supporting evidence if relevant
3. ALWAYS end with ONE relevant follow-up suggestion based on:
   - Related topics mentioned in the cited interviews
   - Connected projects or activities
   - Key people referenced
   - Timeline connections
   Format suggestion as: "Would you like to know more about [specific related topic/person/project]?"

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