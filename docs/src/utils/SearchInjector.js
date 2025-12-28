import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
import CustomSearch from '@site/src/components/CustomSearch';

// Script to render CustomSearch into the HTML container
const SearchContainer = () => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (isClient && typeof document !== 'undefined') {
    const searchContainer = document.getElementById('custom-search-container');
    if (searchContainer) {
      // Create a wrapper div for the search component
      const wrapper = document.createElement('div');
      searchContainer.appendChild(wrapper);

      // Render the CustomSearch component into the wrapper
      ReactDOM.render(<CustomSearch />, wrapper);

      // Clean up function
      return () => {
        if (wrapper.parentNode) {
          wrapper.parentNode.removeChild(wrapper);
        }
      };
    }
  }

  return null; // This component doesn't render anything itself
};

// Only render on client-side
const SearchInjector = () => {
  return ExecutionEnvironment.canUseDOM ? <SearchContainer /> : null;
};

export default SearchInjector;