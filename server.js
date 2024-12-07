const express = require('express');
const cors = require('cors');
const corsOptions = {
	origin: 'https://chatbot11directanswers.netlify.app',
	methods: ['GET', 'POST'],
	credentials: true
};
const dotenv = require('dotenv');
const OpenAI = require('openai');
const path = require('path');
const fs = require('fs');
const csv = require('csv-parse/sync');

// Load environment variables
dotenv.config();

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
            // Load embeddings.json from the backend directory
            const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'embeddings.json'), 'utf8'));
            this.embeddings = data.embeddings;
            this.texts = data.texts;
            this.chunkMetadata = data.metadata;

            // Add logging to see text content
            console.log("First few texts entries:");
            this.texts.slice(0, 3).forEach((text, i) => {
                console.log(`Text ${i + 1} starts with: ${text.substring(0, 100)}...`);
            });

            // Load and parse metadata CSV
            const metadataFile = fs.readFileSync(path.join(__dirname, 'metadata.csv'), 'utf8');
            const records = csv.parse(metadataFile, {
                columns: true,
                skip_empty_lines: true
            });
            
            // Add logging to see metadata records
            console.log("\nFirst few metadata records:");
            records.slice(0, 3).forEach((record, i) => {
                console.log(`Record ${i + 1}: ${record.name}`);
            });

            // Create a map of metadata for efficient lookups
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

    findPersonByName(question) {
        const normalizedQuestion = question.toLowerCase();
        
        // Iterate through metadata to find name matches
        for (const [id, record] of this.metadata.entries()) {
            const normalizedName = record.name.toLowerCase();
            
            // Check if the name appears in the question
            if (normalizedQuestion.includes(normalizedName)) {
                return { id, record };
            }
            
            // Handle possible name variations
            const nameParts = normalizedName.split(' ');
            for (const part of nameParts) {
                if (part.length > 2 && normalizedQuestion.includes(part)) {
                    return { id, record };
                }
            }
        }
        return null;
    }

    async findRelevantContext(question) {
        try {
            // First check if the question is asking about a specific person
            const nameMatch = this.findPersonByName(question);
            if (nameMatch) {
                const { id, record } = nameMatch;
                console.log(`Found match for person: ${record.name} (ID: ${id})`);
                
                const documentName = `document${id}.pdf`;
                const questionEmbedding = await this.getEmbedding(question);
                
                // Find similar content from this document
                const similarContent = await this.findSimilarContent(questionEmbedding, documentName);
                
                if (similarContent.length > 0) {
                    console.log(`Found ${similarContent.length} relevant chunks for document: ${documentName}`);
                    const relevantText = similarContent
                        .map(item => item.text)
                        .join('\n\n');
                    return `Interview ${id} with ${record.name} (${record.date}): ${relevantText}`;
                }
            }

            // Check for explicit interview number mentions
            const interviewMatch = question.match(/interview (?:number )?(\d+)/i);
            if (interviewMatch) {
                const interviewNum = interviewMatch[1];
                const record = this.metadata.get(interviewNum);
                
                if (record) {
                    const documentName = `document${interviewNum}.pdf`;
                    const questionEmbedding = await this.getEmbedding(question);
                    const similarContent = await this.findSimilarContent(questionEmbedding, documentName);
                    
                    if (similarContent.length > 0) {
                        const relevantText = similarContent
                            .map(item => item.text)
                            .join('\n\n');
                        return `Interview ${interviewNum} with ${record.name} (${record.date}): ${relevantText}`;
                    }
                }
            }

            // Check for metadata-specific queries
            const metadataPattern = /(?:who|what|when|where|which|how|tell me about)\s+.*?(name|date|title|tags|interview)/i;
            const isMetadataQuery = metadataPattern.test(question);
            if (isMetadataQuery) {
                const metadataContext = this.searchMetadata(question);
                if (metadataContext) return metadataContext;
            }

            // Default to similarity search with grouping by document
            const questionEmbedding = await this.getEmbedding(question);
            const similarContent = await this.findSimilarContent(questionEmbedding, null);
            
            // Group results by source document
            const groupedResults = {};
            similarContent.forEach(item => {
                const source = item.metadata.source;
                if (!groupedResults[source]) {
                    groupedResults[source] = [];
                }
                groupedResults[source].push(item);
            });

            // Use the document with the most matches
            const bestSource = Object.entries(groupedResults)
                .sort((a, b) => b[1].length - a[1].length)[0];
                
            const relevantChunks = bestSource[1];
            return relevantChunks.map(item => item.text).join('\n\n');

        } catch (error) {
            console.error('Error finding relevant context:', error);
            throw error;
        }
    }

    searchMetadata(question) {
        const q = question.toLowerCase();
        let searchTerms = [];
        
        if (q.includes('interview') || q.includes('interviewed')) {
            searchTerms.push('date');
        }
        if (q.includes('name') || q.includes('who')) {
            searchTerms.push('name');
        }
        if (q.includes('title') || q.includes('about what')) {
            searchTerms.push('excerpt_title');
        }
        if (q.includes('tags') || q.includes('topics')) {
            searchTerms.push('tags');
        }
        let results = [];
        
        Array.from(this.metadata.entries()).forEach(([id, record]) => {
            let matched = false;
            if (searchTerms.length > 0) {
                for (let term of searchTerms) {
                    if (record[term] && record[term].toLowerCase().includes(q)) {
                        matched = true;
                        break;
                    }
                }
            } else {
                const allText = `${record.name} ${record.date} ${record.excerpt_title} ${record.tags}`.toLowerCase();
                matched = allText.includes(q);
            }
            
            if (matched) {
                results.push(`Interview ${id}: ${record.name}, interviewed on ${record.date}. Title: "${record.excerpt_title}". Topics: ${record.tags}`);
            }
        });
        return results.length > 0 ? results.join('\n\n') : null;
    }

    async getEmbedding(text) {
        const response = await this.openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text,
        });
        return response.data[0].embedding;
    }

    async findSimilarContent(queryEmbedding, sourceName = null) {
        // Calculate similarities for all embeddings
        const similarities = this.embeddings.map((emb, idx) => ({
            score: this.cosineSimilarity(queryEmbedding, emb),
            text: this.texts[idx],
            metadata: this.chunkMetadata[idx]
        }));

        // If sourceName is provided, filter for only that document
        let results = similarities;
        if (sourceName) {
            results = similarities.filter(item => item.metadata.source === sourceName);
        }

        // Sort by similarity score and take top 5
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
}

function ensureCompleteResponse(text) {
    // Remove any trailing ellipsis
    text = text.replace(/\.{3,}$/, '');
    
    // Define sentence ending punctuation
    const sentenceEndings = ['.', '!', '?'];
    
    // If the text already ends with proper punctuation, return it
    if (sentenceEndings.some(ending => text.endsWith(ending))) {
        return text;
    }

    // Find the last complete sentence
    let lastCompleteIndex = -1;
    for (const ending of sentenceEndings) {
        const index = text.lastIndexOf(ending);
        if (index > lastCompleteIndex) {
            lastCompleteIndex = index;
        }
    }

    // If we found a complete sentence, trim to that point
    if (lastCompleteIndex !== -1) {
        text = text.substring(0, lastCompleteIndex + 1);
    } else {
        // If no complete sentence is found, append a period if the text isn't empty
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

app.get('/api/test', (req, res) => {
    res.json({ message: 'Server is running!' });
});

app.post('/api/chat', async (req, res) => {
    try {
        const { question, sessionId } = req.body;
        
        if (!question) {
            return res.status(400).json({ error: 'Question is required' });
        }

        let sessionHistory = sessions.get(sessionId) || [];
        const relevantContext = await searchEngine.findRelevantContext(question);

        const completion = await openai.chat.completions.create({
            model: "gpt-4-turbo-preview",
            messages: [
                {
                    role: "system",
                    content: `You are a helpful assistant analyzing oral history interviews. Follow these rules strictly: 

CRITICAL RULES:
- Give complete answers but be extremely concise
- Focus only on answering exactly what was asked
- Skip any background or additional context
- Use simple, direct language
- Stop once you've answered the core question
- Don't try to cover everything from the interview

Example good response:
"Jean Carlomusto primarily worked on AIDS education videos at GMHC and later stepped back from activism in 1993 due to burnout."

Example bad response:
"In her December 2002 interview, Jean Carlomusto discussed her extensive work in AIDS activism. Over the years, she was involved in various projects..."

Only use information from the provided context. If the answer isn't in the context, say "I don't find that information in the interview."

Here is the relevant context:\n\n${relevantContext}`
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
        
        // Post-process the response to ensure complete sentences
        response = ensureCompleteResponse(response);

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

app.use((req, res) => {
    res.status(404).json({ error: 'Route not found' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop the server');
});

process.on('unhandledRejection', (error) => {
    console.error('Unhandled Promise Rejection:', error);
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});