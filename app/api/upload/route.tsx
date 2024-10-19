import { NextRequest, NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import { join } from 'path';
import * as fs from 'fs/promises';

export async function POST(request: NextRequest) {
  try {
    const data = await request.formData();
    const files: File[] = data.getAll('files') as File[];

    if (files.length === 0) {
      return NextResponse.json({ success: false, error: 'No files uploaded' }, { status: 400 });
    }

    const uploadDir = join(process.cwd(), 'public', 'uploads');
    await ensureDir(uploadDir);

    const uploadedFiles = await Promise.all(
      files.map(async (file) => {
        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);

        const path = join(uploadDir, file.name);
        await writeFile(path, buffer);
        return file.name;
      })
    );

    console.log(`Files saved successfully: ${uploadedFiles.join(', ')}`);
    return NextResponse.json({ success: true, uploadedFiles });
  } catch (error) {
    console.error('Error in file upload:', error);
    return NextResponse.json({ success: false, error: 'Server error during upload' }, { status: 500 });
  }
}

async function ensureDir(dirPath: string) {
  try {
    await fs.access(dirPath);
  } catch (error) {
    await fs.mkdir(dirPath, { recursive: true });
  }
}