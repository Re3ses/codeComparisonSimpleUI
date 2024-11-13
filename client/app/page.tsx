'use client';

import React from 'react';
import CodeComparisonApp from './comparator/codeComparison';

export default function Home() {  
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen min-w-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-row row-start-2 items-center justify-center sm:items-start w-full h-full">
        <CodeComparisonApp />
      </main>
    </div>
  );
}
