/**
 * Test script to verify frontend structure and logic
 */

const fs = require('fs');
const path = require('path');

function testFileExists(filePath, description) {
  try {
    const fullPath = path.join(__dirname, filePath);
    if (fs.existsSync(fullPath)) {
      console.log(`[OK] ${description} exists`);
      return true;
    } else {
      console.log(`[FAIL] ${description} NOT found at ${filePath}`);
      return false;
    }
  } catch (e) {
    console.log(`[FAIL] ${description} check failed: ${e.message}`);
    return false;
  }
}

function testFileContent(filePath, patterns, description) {
  try {
    const fullPath = path.join(__dirname, filePath);
    const content = fs.readFileSync(fullPath, 'utf-8');
    
    for (const pattern of patterns) {
      if (content.includes(pattern)) {
        console.log(`[OK] ${description}: Found "${pattern}"`);
      } else {
        console.log(`[FAIL] ${description}: Missing "${pattern}"`);
        return false;
      }
    }
    return true;
  } catch (e) {
    console.log(`[FAIL] ${description} check failed: ${e.message}`);
    return false;
  }
}

console.log('='.repeat(50));
console.log('Frontend Structure Tests');
console.log('='.repeat(50));

const results = [];

// Test file existence
console.log('\nTesting file existence...');
results.push(['API Client', testFileExists('src/lib/api.ts', 'API client')]);
results.push(['Types', testFileExists('src/lib/types.ts', 'Types file')]);
results.push(['Chat Page', testFileExists('src/app/chat/page.tsx', 'Chat page')]);
results.push(['Home Page', testFileExists('src/app/page.tsx', 'Home page')]);
results.push(['Source Input Dialog', testFileExists('src/components/chat/source-input-dialog.tsx', 'Source input dialog')]);

// Test API client structure
console.log('\nTesting API client structure...');
results.push(['API Client Methods', testFileContent('src/lib/api.ts', [
  'signup',
  'signin',
  'getChats',
  'createChat',
  'getChatMessages',
  'streamChat',
  'createYouTubeRAG',
  'createPDFRAG',
  'createWebRAG',
  'createGitRAG'
], 'API client methods')]);

results.push(['API Client Token Management', testFileContent('src/lib/api.ts', [
  'getToken',
  'setToken',
  'removeToken',
  'localStorage'
], 'API client token management')]);

// Test types
console.log('\nTesting types...');
results.push(['Chat Type', testFileContent('src/lib/types.ts', [
  'ChatType',
  'vector_db_collection_id'
], 'Chat type definition')]);

// Test chat page integration
console.log('\nTesting chat page integration...');
results.push(['Chat Page API Calls', testFileContent('src/app/chat/page.tsx', [
  'api.getChats',
  'api.createChat',
  'api.getChatMessages',
  'api.streamChat'
], 'Chat page API integration')]);

// Test authentication integration
console.log('\nTesting authentication integration...');
results.push(['Home Page Auth', testFileContent('src/app/page.tsx', [
  'api.signup',
  'api.signin'
], 'Home page authentication')]);

// Test RAG integration
console.log('\nTesting RAG integration...');
results.push(['RAG Source Creation', testFileContent('src/app/chat/page.tsx', [
  'createYouTubeRAG',
  'createPDFRAG',
  'createWebRAG',
  'createGitRAG'
], 'RAG source creation')]);

// Test source input dialog
console.log('\nTesting source input dialog...');
results.push(['Source Input File Handling', testFileContent('src/components/chat/source-input-dialog.tsx', [
  'File',
  'onSubmit'
], 'Source input file handling')]);

console.log('\n' + '='.repeat(50));
console.log('Test Results Summary');
console.log('='.repeat(50));

const passed = results.filter(([_, result]) => result).length;
const total = results.length;

results.forEach(([name, result]) => {
  const status = result ? '[PASS]' : '[FAIL]';
  console.log(`${status}: ${name}`);
});

console.log(`\nTotal: ${passed}/${total} tests passed`);

if (passed === total) {
  console.log('\n[SUCCESS] All frontend structure tests passed!');
  process.exit(0);
} else {
  console.log('\n[ERROR] Some tests failed');
  process.exit(1);
}









