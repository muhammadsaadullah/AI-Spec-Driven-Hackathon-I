// Client-side search injector for Docusaurus
document.addEventListener('DOMContentLoaded', function() {
  // Wait a bit for the navbar to be fully rendered
  setTimeout(function() {
    const searchContainer = document.getElementById('custom-search-container');
    if (searchContainer) {
      // Create the search input and button
      const searchWrapper = document.createElement('div');
      searchWrapper.className = 'search-wrapper';
      searchWrapper.style.display = 'flex';
      searchWrapper.style.alignItems = 'center';
      searchWrapper.style.minWidth = '200px';

      const searchInput = document.createElement('input');
      searchInput.type = 'text';
      searchInput.placeholder = 'Search...';
      searchInput.style.padding = '8px 12px';
      searchInput.style.border = '1px solid #ccc';
      searchInput.style.borderRadius = '4px 0 0 4px';
      searchInput.style.outline = 'none';
      searchInput.style.width = '150px';
      searchInput.style.fontSize = '14px';

      const searchButton = document.createElement('button');
      searchButton.innerHTML = '🔍';
      searchButton.style.background = '#25c2a0';
      searchButton.style.color = 'white';
      searchButton.style.border = 'none';
      searchButton.style.borderRadius = '0 4px 4px 0';
      searchButton.style.padding = '8px 12px';
      searchButton.style.cursor = 'pointer';
      searchButton.style.fontSize = '14px';

      searchWrapper.appendChild(searchInput);
      searchWrapper.appendChild(searchButton);
      searchContainer.appendChild(searchWrapper);

      // Add basic search functionality
      searchInput.addEventListener('keyup', function(event) {
        if (event.key === 'Enter') {
          performSearch(searchInput.value);
        }
      });

      searchButton.addEventListener('click', function() {
        performSearch(searchInput.value);
      });

      function performSearch(query) {
        if (query.trim()) {
          // For now, just show an alert - in the future this could call the API
          alert('Search for: ' + query);
        }
      }
    }
  }, 500);
});