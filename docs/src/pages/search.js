import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import ClientOnly from '@docusaurus/ClientOnly';
import CustomSearch from '@site/src/components/CustomSearch';

// Search page component that renders our custom search
function SearchPage() {
  return (
    <Layout title="Search" description="Search the Physical AI & Humanoid Robotics Course">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Search Course Content</h1>
            <ClientOnly>
              {() => <CustomSearch />}
            </ClientOnly>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default SearchPage;