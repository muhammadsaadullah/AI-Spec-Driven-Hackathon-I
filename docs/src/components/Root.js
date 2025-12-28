import React from 'react';
import SearchInjector from './utils/SearchInjector';

// Root component that wraps the entire app
const Root = ({ children }) => {
  return (
    <>
      <SearchInjector />
      {children}
    </>
  );
};

export default Root;