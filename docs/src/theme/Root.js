import React, { useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import CustomSearch from '../components/CustomSearch';

// Root component to inject our custom search functionality
const Root = ({ children }) => {
  useEffect(() => {
    let searchRoot;

    const initializeSearch = async () => {
      // Wait for the DOM to be ready and the search container to exist
      const waitForContainer = () => {
        return new Promise((resolve) => {
          const checkContainer = () => {
            const container = document.getElementById('custom-search-container');
            if (container) {
              resolve(container);
            } else {
              setTimeout(checkContainer, 100);
            }
          };
          checkContainer();
        });
      };

      try {
        const container = await waitForContainer();

        // Create a wrapper element for our search component
        const wrapper = document.createElement('div');
        wrapper.style.display = 'inline-block';
        wrapper.style.verticalAlign = 'middle';
        container.appendChild(wrapper);

        // Create React root and render the search component
        searchRoot = ReactDOM.createRoot(wrapper);
        searchRoot.render(<CustomSearch />);
      } catch (error) {
        console.error('Error initializing custom search:', error);
      }
    };

    initializeSearch();

    // Cleanup function
    return () => {
      if (searchRoot) {
        searchRoot.unmount();
      }
    };
  }, []);

  return <>{children}</>;
};

export default Root;