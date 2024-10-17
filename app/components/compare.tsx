'use client';

import React, { useState, useEffect } from 'react';

export default function CompareFiles() {
  const [files, setFiles] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [compareResult, setCompareResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await fetch('/api/files');
      if (response.ok) {
        const data = await response.json();
        setFiles(data.files);
      }
    } catch (error) {
      console.error('Error fetching files:', error);
      setError('Failed to fetch files. Please try again.');
    }
  };

  const handleCompare = async () => {
    setIsLoading(true);
    setError(null);
    setCompareResult(null);

    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ files }),
      });

      if (response.ok) {
        const result = await response.json();
        setCompareResult(result.comparisonResult);
      } else {
        throw new Error('Comparison failed');
      }
    } catch (error) {
      console.error('Error comparing files:', error);
      setError('Failed to compare files. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className='my-2 border-2 p-3 rounded-lg w-full flex flex-col'>
      <div className='border-b-2 pb-2 w-full flex flex-row justify-between items-center'>
      <h2 className='text-2xl font-bold'>Compare Uploaded Files</h2>
      <button onClick={handleCompare} disabled={isLoading || files.length === 0} className="px-5 py-2 bg-white hover:bg-slate-400 rounded-full text-black">
        {isLoading ? 'Comparing...' : 'Compare Files'}
      </button>
      </div>
      <div className='mt-2'>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {compareResult && (
        <div>
          <h3>Comparison Result:</h3>
          <pre>{compareResult}</pre>
        </div>
      )}
      </div>
    </div>
  );
}