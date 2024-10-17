'use client';

import React, { useState } from 'react';

interface FileUploadFormProps {
  onUploadSuccess: () => void;
}

export default function FileUploadForm({ onUploadSuccess }: FileUploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
      setError(null);
      setSuccess(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        console.log('File uploaded successfully');
        setSuccess(true);
        setError(null);
        onUploadSuccess(); // Call the callback function
      } else {
        console.error('File upload failed:', result.error);
        setError(result.error || 'File upload failed');
        setSuccess(false);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setError('Error uploading file. Please try again.');
      setSuccess(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className='border-2 p-3 rounded-lg w-full flex flex-row justify-between items-center'>
      <input type="file" onChange={handleFileChange} className='bg-slate-50 text-black rounded-lg h-fit w-fit'/>
      <button type="submit" className="px-5 py-2 bg-white hover:bg-slate-400 rounded-full text-black">Upload</button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {success && <p style={{ color: 'green' }}>File uploaded successfully!</p>}
    </form>
  );
}