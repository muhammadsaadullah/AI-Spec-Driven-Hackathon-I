const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

/**
 * Advanced content extractor for RAG chatbot applications
 * This script extracts, chunks, and formats content for vector database indexing
 */

class RAGContentExtractor {
  constructor(options = {}) {
    this.chunkSize = options.chunkSize || 1000; // characters per chunk
    this.chunkOverlap = options.chunkOverlap || 200; // overlap between chunks
    this.outputDir = options.outputDir || './rag-data';
  }

  /**
   * Split content into overlapping chunks for better RAG performance
   */
  chunkText(text, chunkSize, overlap) {
    const chunks = [];
    let start = 0;

    while (start < text.length) {
      let end = start + chunkSize;

      // If we're not at the end and there's a space nearby, break at the space
      if (end < text.length) {
        const nextSpace = text.lastIndexOf(' ', end);
        if (nextSpace > start + chunkSize / 2) {
          end = nextSpace;
        }
      }

      const chunk = text.slice(start, end).trim();
      if (chunk) {
        chunks.push(chunk);
      }

      start = end - overlap;
    }

    return chunks;
  }

  /**
   * Extract and process all documentation content
   */
  extractDocsContent(docsDir = './docs') {
    const chunks = [];
    const documents = [];

    // Read all markdown files in the docs directory
    const walkSync = (dir, filelist = []) => {
      const files = fs.readdirSync(dir);

      files.forEach(file => {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
          filelist = walkSync(filePath, filelist);
        } else if ((file.endsWith('.md') || file.endsWith('.mdx')) && !file.startsWith('_')) {
          filelist.push(filePath);
        }
      });

      return filelist;
    };

    const markdownFiles = walkSync(docsDir);

    markdownFiles.forEach((file, index) => {
      try {
        const fileContent = fs.readFileSync(file, 'utf8');
        const parsed = matter(fileContent);

        // Extract content and metadata
        const relativePath = path.relative(docsDir, file);
        const slug = relativePath.replace(/\\/g, '/').replace('.md', '').replace('.mdx', '');
        const url = `/docs/${slug}`;

        // Create document record
        const document = {
          id: `doc_${index}`,
          title: parsed.data.title || 'Untitled',
          source: relativePath,
          url: url,
          metadata: {
            ...parsed.data,
            path: slug,
            sourceFile: file,
            createdAt: fs.statSync(file).mtime,
          }
        };

        documents.push(document);

        // Split content into chunks
        const contentChunks = this.chunkText(parsed.content, this.chunkSize, this.chunkOverlap);

        contentChunks.forEach((chunk, chunkIndex) => {
          chunks.push({
            id: `chunk_${index}_${chunkIndex}`,
            documentId: `doc_${index}`,
            title: parsed.data.title || 'Untitled',
            content: chunk,
            source: relativePath,
            url: url,
            metadata: {
              ...parsed.data,
              path: slug,
              chunkIndex,
              sourceFile: file,
              createdAt: fs.statSync(file).mtime,
            }
          });
        });

      } catch (error) {
        console.warn(`Error processing file ${file}:`, error.message);
      }
    });

    return { documents, chunks };
  }

  /**
   * Save extracted content to JSON files
   */
  saveExtractedContent({ documents, chunks }, outputDir = this.outputDir) {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    fs.writeFileSync(path.join(outputDir, 'documents.json'), JSON.stringify(documents, null, 2));
    fs.writeFileSync(path.join(outputDir, 'chunks.json'), JSON.stringify(chunks, null, 2));

    console.log(`Extracted ${documents.length} documents and ${chunks.length} chunks`);
    console.log(`Saved to ${outputDir}/`);
  }

  /**
   * Generate a sample integration file for your RAG system
   */
  generateRAGIntegrationSample(outputDir = this.outputDir) {
    const sampleCode = `// Sample integration code for your RAG chatbot
const fs = require('fs');

// Load the extracted content
const chunks = JSON.parse(fs.readFileSync('${outputDir}/chunks.json', 'utf8'));

// Example: Find relevant chunks for a query
function findRelevantChunks(query, topK = 5) {
  // This is where you'd implement vector similarity search
  // using your chosen vector database (Pinecone, Weaviate, etc.)

  // For now, this is a simple keyword match as a placeholder
  return chunks
    .filter(chunk =>
      chunk.content.toLowerCase().includes(query.toLowerCase()) ||
      chunk.title.toLowerCase().includes(query.toLowerCase())
    )
    .slice(0, topK);
}

// Example usage
const query = "your question here";
const relevantChunks = findRelevantChunks(query);
console.log("Relevant content for RAG:", relevantChunks);

module.exports = { findRelevantChunks, chunks };
`;

    fs.writeFileSync(path.join(outputDir, 'rag-integration-sample.js'), sampleCode);
    console.log(`Generated RAG integration sample at ${outputDir}/rag-integration-sample.js`);
  }

  /**
   * Execute the full extraction process
   */
  extractAndSave(docsDir = './docs', outputDir = this.outputDir) {
    console.log('Starting content extraction for RAG system...');
    const result = this.extractDocsContent(docsDir);
    this.saveExtractedContent(result, outputDir);
    this.generateRAGIntegrationSample(outputDir);
    console.log('Content extraction completed successfully!');
    return result;
  }
}

// If running as a script
if (require.main === module) {
  const extractor = new RAGContentExtractor({
    chunkSize: 1000,
    chunkOverlap: 200,
    outputDir: './rag-data'
  });

  extractor.extractAndSave('./docs');
}

module.exports = RAGContentExtractor;