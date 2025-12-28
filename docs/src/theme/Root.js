import React from 'react';

// Simple Root component that doesn't interfere with SSR
const Root = ({ children }) => {
  return <>{children}</>;
};

export default Root;