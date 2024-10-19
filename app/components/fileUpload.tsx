'use client';

import React, { useState } from 'react';

interface FileUploadFormProps {
  onUploadSuccess: () => void;
}

export default function FileUploadForm({ onUploadSuccess }: FileUploadFormProps) {
  const [files, setFiles] = useState<FileList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(e.target.files);
      setError(null);
      setSuccess(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!files || files.length === 0) {
      setError('Please select at least one file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    Array.from(files).forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        console.log('Files uploaded successfully');
        setSuccess(true);
        setError(null);
        onUploadSuccess();
      } else {
        console.error('File upload failed:', result.error);
        setError(result.error || 'File upload failed');
        setSuccess(false);
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      setError('Error uploading files. Please try again.');
      setSuccess(false);
    } finally {
      setUploading(false);
    }
  };


  return (
    <form onSubmit={handleSubmit} className='border-2 p-3 rounded-lg w-full flex flex-row justify-between items-center'>
      <input type="file" onChange={handleFileChange} className='bg-slate-50 text-black rounded-lg h-fit w-fit' multiple/>
      <button type="submit" className="px-5 py-2 bg-white hover:bg-slate-400 rounded-full text-black">Upload</button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {success && <p style={{ color: 'green' }}>File uploaded successfully!</p>}
    </form>
  );
}