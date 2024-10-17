import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    const { files } = await request.json();

    if (!Array.isArray(files) || files.length === 0) {
      return NextResponse.json({ error: 'No files provided for comparison' }, { status: 400 });
    }

    const uploadDir = path.join(process.cwd(), 'public', 'uploads');
    const fileContents = await Promise.all(
      files.map(async (file) => {
        const filePath = path.join(uploadDir, file);
        const content = await fs.readFile(filePath, 'utf-8');
        return { name: file, content };
      })
    );

    // ========================================
    // Implement your comparison logic here
    // ========================================

    // This is a simple example that just concatenates the file contents
    const comparisonResult = fileContents
      .map(({ name, content }) => `File: ${name}\n${content}\n\n`)
      .join('');

    return NextResponse.json({ comparisonResult });
  } catch (error) {
    console.error('Error in file comparison:', error);
    return NextResponse.json({ error: 'Server error during comparison' }, { status: 500 });
  }
}