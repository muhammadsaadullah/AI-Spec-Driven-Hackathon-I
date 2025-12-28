import React, { useEffect } from 'react';

// SSR-safe Root component for search injection
const Root = ({ children }) => {
  useEffect(() => {
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

        // Dynamically load and render the CustomSearch component
        const { default: CustomSearch } = await import('../components/CustomSearch');

        // Create a wrapper element for our search component
        const wrapper = document.createElement('div');
        wrapper.style.display = 'inline-block';
        wrapper.style.verticalAlign = 'middle';
        wrapper.style.minWidth = '200px'; // Ensure space for search
        container.appendChild(wrapper);

        // Render the component using React DOM
        const React = await import('react');
        const ReactDOM = await import('react-dom/client');

        if (ReactDOM.createRoot) {
          const root = ReactDOM.createRoot(wrapper);
          root.render(React.createElement(CustomSearch));
        }
      } catch (error) {
        console.error('Error initializing search component:', error);
      }
    };

    initializeSearch();
  }, []);

  return <>{children}</>;
};

export default Root;