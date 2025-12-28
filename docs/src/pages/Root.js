import React, { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import CustomSearch from '@site/src/components/CustomSearch';

// Root component that will handle the custom search injection
const Root = ({ children }) => {
  useEffect(() => {
    // Find the search container and render our custom search there
    const searchContainer = document.getElementById('custom-search-container');
    if (searchContainer) {
      // Create a react root for the search component
      const searchRoot = createRoot(searchContainer);
      searchRoot.render(<CustomSearch />);

      // Cleanup function
      return () => {
        searchRoot.unmount();
      };
    }
  }, []);

  return <>{children}</>;
};

export default Root;