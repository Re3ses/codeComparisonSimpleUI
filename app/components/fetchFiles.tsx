'use client';

import React, { useState, useEffect } from 'react';

interface FileListProps {
  refreshTrigger: number;
}

export default function FileList({ refreshTrigger }: FileListProps) {
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

  const handleDelete = async (fileName: string) => {
    try {
      const response = await fetch(`/api/files?file=${fileName}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        console.log('File deleted successfully');
        fetchFiles();
      } else {
        throw new Error('Failed to delete file');
      }
    } catch (error) {
      console.error('Error deleting file:', error);
    }
  }

  useEffect(() => {
    fetchFiles();
  }, [refreshTrigger]);

  return (
    <div className='flex flex-col w-full'>
      <div className='flex flex-row justify-between w-full'>
        <h2 className='font-bold text-2xl'>Uploaded Files</h2>
        <button onClick={fetchFiles} className='px-3 py-1 bg-white hover:bg-slate-400 rounded-full text-black my-2 text-sm'>Refresh File List</button>
      </div>
      <ul className='m-2 border-solid border-2 p-3 rounded-lg'>
        {files.map((file, index) => (
          <li key={index} className='border-b-2 py-1 flex flex-row justify-between items-center'>
            <a href={`/uploads/${file}`} target="_blank" rel="noopener noreferrer" className='w-full hover:bg-gray-600'>
              {file}
            </a>
            <button type="button" onClick={() => handleDelete(file)} className='text-white px-2 border-2 rounded-lg h-full'>x</button>
          </li>
        ))}
      </ul>
    </div>
  );
}