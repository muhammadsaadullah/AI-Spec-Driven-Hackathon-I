import React, { useEffect } from 'react';
import CustomSearch from './CustomSearch';

// Test component to ensure search integration works
const TestSearchIntegration = () => {
  useEffect(() => {
    // Check if the custom search container exists
    const container = document.getElementById('custom-search-container');
    if (container && container.children.length === 0) {
      // If no React component has been mounted yet, try to render our search component
      const wrapper = document.createElement('div');
      container.appendChild(wrapper);

      // Try to dynamically render the CustomSearch component
      const renderSearch = async () => {
        try {
          // This approach might work better with Docusaurus
          const CustomSearchModule = await import('./CustomSearch');
          const CustomSearchComponent = CustomSearchModule.default;

          // Use React to render the component into the wrapper
          const React = await import('react');
          const ReactDOM = await import('react-dom/client');

          if (CustomSearchComponent && ReactDOM.createRoot) {
            const root = ReactDOM.createRoot(wrapper);
            root.render(React.createElement(CustomSearchComponent));
          }
        } catch (error) {
          console.error('Error rendering search component:', error);

          // Fallback: Add a simple placeholder to indicate search should be here
          wrapper.innerHTML = '<div style="display: inline-block; padding: 8px 12px; font-size: 14px; background: #f0f0f0; border-radius: 4px;">🔍 Search</div>';
        }
      };

      renderSearch();
    }
  }, []);

  // This component doesn't render anything itself
  return null;
};

export default TestSearchIntegration;