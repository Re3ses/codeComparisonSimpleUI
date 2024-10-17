'use client';

import React, { useState } from 'react';
import FileUploadForm from './components/fileUpload';
import FileList from './components/fetchFiles';
import CompareFiles from './components/compare';

export default function Home() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen min-w-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-row gap-8 row-start-2 items-center sm:items-start w-full h-full">
        <div className="flex flex-col w-1/2">
          <FileUploadForm onUploadSuccess={handleUploadSuccess} />
          <CompareFiles />
        </div>
        <div className="flex w-1/2">
          <FileList key={refreshTrigger} />
        </div>
      </main>
    </div>
  );
}
