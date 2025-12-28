import React from 'react';

// Root component to inject our custom search functionality
const Root = ({ children }) => {
  return <>{children}</>;
};

// Initialize search functionality only on the client-side
if (typeof window !== 'undefined') {
  // Client-side code only
  const initializeSearch = async () => {
    // Wait for the DOM to be ready
    if (document.readyState === 'loading') {
      await new Promise(resolve => {
        document.addEventListener('DOMContentLoaded', resolve);
      });
    }

    // Find the search container element
    const container = document.getElementById('custom-search-container');

    if (container) {
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
  initializeSearch();
}

export default Root;