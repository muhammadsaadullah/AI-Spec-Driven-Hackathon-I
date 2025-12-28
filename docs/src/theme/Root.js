import React from 'react';

// Simple Root component that doesn't do anything during SSR
const Root = ({ children }) => {
  return <>{children}</>;
};

export default Root;