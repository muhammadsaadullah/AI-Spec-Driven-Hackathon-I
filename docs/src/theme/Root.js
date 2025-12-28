import React from 'react';

// Root component to inject our custom search functionality
const Root = ({ children }) => {
  // SSR check - only run client-side code
  if (typeof window !== 'undefined') {
    // Client-side code only
    const initializeSearch = async () => {
      // Find the search container element
      const container = document.getElementById('custom-search-container');

      if (container && !container.hasChildNodes()) {
        // Dynamically import the CustomSearch component
        const { default: CustomSearch } = await import('../components/CustomSearch');

        // Create a wrapper element for our search component
        const wrapper = document.createElement('div');
        wrapper.style.display = 'inline-block';
        wrapper.style.verticalAlign = 'middle';
        wrapper.style.minWidth = '200px'; // Ensure space for search
        container.appendChild(wrapper);

        // Use React DOM to render the component
        const React = await import('react');
        const ReactDOM = await import('react-dom/client');

        if (ReactDOM.createRoot) {
          const root = ReactDOM.createRoot(wrapper);
          root.render(React.createElement(CustomSearch));
        } else {
          // Fallback for older versions
          const ReactDOMLegacy = await import('react-dom');
          ReactDOMLegacy.render(React.createElement(CustomSearch), wrapper);
        }
      }
    };

    // Initialize search after DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initializeSearch);
    } else {
      initializeSearch();
    }
  }

  return <>{children}</>;
};

export default Root;