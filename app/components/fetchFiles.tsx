'use client';

import React, { useState, useEffect } from 'react';

export default function FileList() {
  const [files, setFiles] = useState<string[]>([]);

  const fetchFiles = async () => {
    try {
      const response = await fetch('/api/files');
      if (response.ok) {
        const data = await response.json();
        setFiles(data.files);
      }
    } catch (error) {
      console.error('Error fetching files:', error);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  return (
    <div className='flex flex-col w-full'>
      <div className='flex flex-row justify-between w-full'>
      <h2 className='font-bold text-2xl'>Uploaded Files</h2>
      <button onClick={fetchFiles} className='px-3 py-1 bg-white hover:bg-slate-400 rounded-full text-black my-2 text-sm'>Refresh File List</button>
      </div>
      <ul className='m-2 border-solid border-2 p-3 rounded-lg'>
        {files.map((file, index) => (
          <li key={index} className='border-b-2 py-1 hover:bg-gray-700'>
            <a href={`/uploads/${file}`} target="_blank" rel="noopener noreferrer">
              {file}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}