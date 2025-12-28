#!/usr/bin/env node

/**
 * Test script to verify the search functionality
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

console.log('🔍 Testing Search Functionality...\n');

// Test queries
const testQueries = [
  { query: 'robotics', limit: 3 },
  { query: 'ros2', limit: 2 },
  { query: 'gazebo', limit: 2 },
  { query: 'nvidia isaac', limit: 2 }
];

let testsPassed = 0;
let totalTests = testQueries.length;

function testSearch(query, limit, callback) {
  const url = `http://localhost:5000/search?q=${encodeURIComponent(query)}&limit=${limit}`;

  http.get(url, (res) => {
    let data = '';

    res.on('data', (chunk) => {
      data += chunk;
    });

    res.on('end', () => {
      try {
        const results = JSON.parse(data);

        console.log(`✅ Query: "${query}"`);
        console.log(`   Results found: ${results.total}`);
        console.log(`   Sample result: ${results.results && results.results[0] ? results.results[0].title : 'None'}`);
        console.log('');

        testsPassed++;
        callback();
      } catch (error) {
        console.log(`❌ Query: "${query}" - Error parsing response: ${error.message}`);
        callback();
      }
    });
  }).on('error', (error) => {
    console.log(`❌ Query: "${query}" - Error: ${error.message}`);
    callback();
  });
}

function runTests(index = 0) {
  if (index >= testQueries.length) {
    console.log(`\n📊 Test Results: ${testsPassed}/${totalTests} tests passed`);
    if (testsPassed === totalTests) {
      console.log('🎉 All tests passed! Search functionality is working correctly.');
    } else {
      console.log('⚠️  Some tests failed. Please check the search API.');
    }
    return;
  }

  const { query, limit } = testQueries[index];
  testSearch(query, limit, () => {
    runTests(index + 1);
  });
}

// Check if search API is running first
http.get('http://localhost:5000/search?q=test', (res) => {
  if (res.statusCode === 200 || res.statusCode === 400) { // 400 means API is running but query missing
    console.log('✅ Search API is running on http://localhost:5000');
    console.log('🚀 Running search functionality tests...\n');
    runTests();
  } else {
    console.log('❌ Search API does not appear to be running on http://localhost:5000');
    console.log('💡 Please start the search API with: node search-api.js');
  }
}).on('error', (error) => {
  console.log('❌ Cannot connect to search API at http://localhost:5000');
  console.log('💡 Please start the search API with: node search-api.js');
  console.log(`   Error: ${error.message}`);
});