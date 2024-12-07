// scripts/generateEmbeddings.js
const fs = require('fs');
const path = require('path');
const pdf = require('pdf-parse');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const { encode } = require('gpt-3-encoder');

// Load environment variables
dotenv.config();

class DocumentProcessor {
    constructor() {
        this.openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });
    }

    async processDocuments() {
        try {
            // 1. Read documents from the frontend PDF directory
            const docs = await this.readDocuments();
            console.log(`Found ${docs.length} documents`);
            
            // 2. Split documents into chunks
            const chunks = await this.splitIntoChunks(docs);
            console.log(`Created ${chunks.length} chunks`);

            // 3. Generate embeddings for each chunk
            const embeddings = await this.generateEmbeddings(chunks);
            
            // 4. Save embeddings and chunks
            await this.saveEmbeddings(embeddings, chunks);
            
            console.log('Embedding generation complete!');
        } catch (error) {
            console.error('Error processing documents:', error);
        }
    }

    async readDocuments() {
        // Adjust path to point to frontend PDF directory
        const pdfPath = path.join(__dirname, '..', '..', 'frontend', 'assets', 'pdfs');
        let files = fs.readdirSync(pdfPath);
        
        // Sort files numerically
        files.sort((a, b) => {
            const numA = parseInt(a.match(/\d+/)[0]);
            const numB = parseInt(b.match(/\d+/)[0]);
            return numA - numB;
        });

        const documents = [];

        for (const file of files) {
            if (file.endsWith('.pdf')) {
                const filePath = path.join(pdfPath, file);
                console.log(`Processing ${file}...`);
                
                try {
                    const dataBuffer = fs.readFileSync(filePath);
                    const pdfData = await pdf(dataBuffer);
                    documents.push({
                        text: pdfData.text,
                        title: file,
                        pages: pdfData.numpages
                    });
                    console.log(`Successfully processed ${file} (${pdfData.numpages} pages)`);
                } catch (error) {
                    console.error(`Error processing ${file}:`, error);
                }
            }
        }

        return documents;
    }

    async splitIntoChunks(documents, maxTokens = 500) {
        const chunks = [];

        for (const doc of documents) {
            let currentChunk = '';
            const paragraphs = doc.text.split(/\n\s*\n/); // Split by double newline

            for (const paragraph of paragraphs) {
                const trimmedParagraph = paragraph.trim();
                if (!trimmedParagraph) continue;

                // Check if adding this paragraph would exceed maxTokens
                const potentialChunk = currentChunk + '\n' + trimmedParagraph;
                const tokenCount = encode(potentialChunk).length;

                if (tokenCount > maxTokens && currentChunk) {
                    // Store current chunk and start a new one
                    chunks.push({
                        text: currentChunk.trim(),
                        source: doc.title,
                        tokens: encode(currentChunk).length
                    });
                    currentChunk = trimmedParagraph;
                } else {
                    currentChunk = potentialChunk;
                }
            }

            // Add the last chunk if it's not empty
            if (currentChunk.trim()) {
                chunks.push({
                    text: currentChunk.trim(),
                    source: doc.title,
                    tokens: encode(currentChunk).length
                });
            }
        }

        return chunks;
    }

    async generateEmbeddings(chunks) {
        const embeddings = [];
        const batchSize = 20; // Process in batches to avoid rate limits
        
        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            console.log(`Processing batch ${i / batchSize + 1} of ${Math.ceil(chunks.length / batchSize)}`);
            
            const batchPromises = batch.map(async chunk => {
                try {
                    const response = await this.openai.embeddings.create({
                        model: "text-embedding-3-small",
                        input: chunk.text,
                    });
                    return response.data[0].embedding;
                } catch (error) {
                    console.error(`Error generating embedding for chunk: ${error.message}`);
                    return null;
                }
            });

            const batchEmbeddings = await Promise.all(batchPromises);
            embeddings.push(...batchEmbeddings.filter(emb => emb !== null));
            
            // Add a small delay between batches to avoid rate limits
            if (i + batchSize < chunks.length) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

        return embeddings;
    }

    async saveEmbeddings(embeddings, chunks) {
        // Save in backend directory
        const outputPath = path.join(__dirname, '..', 'embeddings.json');
        
        // Prepare the data structure
        const data = {
            embeddings: embeddings,
            texts: chunks.map(chunk => chunk.text),
            metadata: chunks.map(chunk => ({
                source: chunk.source,
                tokens: chunk.tokens
            }))
        };

        // Save the embeddings
        fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
        console.log(`Saved embeddings to ${outputPath}`);
    }
}

// Run the embedding generation
const processor = new DocumentProcessor();
processor.processDocuments();