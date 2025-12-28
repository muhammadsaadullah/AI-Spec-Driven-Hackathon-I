const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

/**
 * Extract content from Docusaurus docs for vector database indexing
 * This script will extract all markdown content for your RAG chatbot
 */

function extractDocsContent(docsDir = './docs') {
  const content = [];

  // Read all markdown files in the docs directory
  const walkSync = (dir, filelist = []) => {
    const files = fs.readdirSync(dir);

    files.forEach(file => {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);

      if (stat.isDirectory()) {
        filelist = walkSync(filePath, filelist);
      } else if (file.endsWith('.md') || file.endsWith('.mdx')) {
        filelist.push(filePath);
      }
    });

    return filelist;
  };

  const markdownFiles = walkSync(docsDir);

  markdownFiles.forEach(file => {
    try {
      const fileContent = fs.readFileSync(file, 'utf8');
      const parsed = matter(fileContent);

      // Extract content and metadata
      const relativePath = path.relative(docsDir, file);
      const slug = relativePath.replace(/\\/g, '/').replace('.md', '').replace('.mdx', '');

      content.push({
        id: slug,
        title: parsed.data.title || 'Untitled',
        content: parsed.content,
        source: relativePath,
        metadata: {
          ...parsed.data,
          path: slug,
          url: `/docs/${slug}`
        }
      });
    } catch (error) {
      console.warn(`Error processing file ${file}:`, error.message);
    }
  });

  return content;
}

function saveExtractedContent(content, outputPath = './extracted-content.json') {
  fs.writeFileSync(outputPath, JSON.stringify(content, null, 2));
  console.log(`Extracted ${content.length} documents to ${outputPath}`);
}

// Extract content from docs
const docsContent = extractDocsContent('./docs');
saveExtractedContent(docsContent);

module.exports = {
  extractDocsContent,
  saveExtractedContent
};