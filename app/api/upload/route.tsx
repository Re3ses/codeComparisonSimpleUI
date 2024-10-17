import { NextRequest, NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import { join } from 'path';

export async function POST(request: NextRequest) {
  try {
    const data = await request.formData();
    const file: File | null = data.get('file') as unknown as File;

    if (!file) {
      return NextResponse.json({ success: false, error: 'No file uploaded' }, { status: 400 });
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Ensure the uploads directory exists
    const uploadDir = join(process.cwd(), 'public', 'uploads');
    await ensureDir(uploadDir);

    const path = join(uploadDir, file.name);
    await writeFile(path, buffer);

    console.log(`File saved successfully: ${path}`);
    return NextResponse.json({ success: true, fileName: file.name });
  } catch (error) {
    console.error('Error in file upload:', error);
    return NextResponse.json({ success: false, error: 'Server error during upload' }, { status: 500 });
  }
}

// Helper function to ensure a directory exists
async function ensureDir(dirPath: string) {
  try {
    await fs.access(dirPath);
  } catch (error) {
    await fs.mkdir(dirPath, { recursive: true });
  }
}

// Import fs promises at the top of the file
import * as fs from 'fs/promises';