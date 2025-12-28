import React, { useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import CustomSearch from '../components/CustomSearch';

// Root component to inject our custom search functionality
const Root = ({ children }) => {
  const searchContainerRef = useRef(null);
  const searchRootRef = useRef(null);

  useEffect(() => {
    const initializeSearch = async () => {
      // Find the search container element
      const container = document.getElementById('custom-search-container');

      if (container) {
        // If there's an existing search root, unmount it
        if (searchRootRef.current) {
          searchRootRef.current.unmount();
        }

        // Create a wrapper element for our search component
        const wrapper = document.createElement('div');
        wrapper.style.display = 'inline-block';
        wrapper.style.verticalAlign = 'middle';
        wrapper.style.minWidth = '200px'; // Ensure space for search
        container.appendChild(wrapper);

        // Create React root and render the search component
        const root = createRoot(wrapper);
        root.render(<CustomSearch />);

        // Store the root reference
        searchRootRef.current = root;
        searchContainerRef.current = container;
      }
    };

    // Initialize search immediately
    initializeSearch();

    // Also set up a MutationObserver to handle page changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          const container = document.getElementById('custom-search-container');
          if (container && !searchRootRef.current) {
            // Container was added, initialize search
            initializeSearch();
          }
        }
      });
    });

    // Observe for changes to the DOM that might affect the search container
    if (document.body) {
      observer.observe(document.body, {
        childList: true,
        subtree: true,
      });
    }

    // Cleanup function
    return () => {
      if (searchRootRef.current) {
        searchRootRef.current.unmount();
        searchRootRef.current = null;
      }
      observer.disconnect();
    };
  }, []);

  return <>{children}</>;
};

export default Root;