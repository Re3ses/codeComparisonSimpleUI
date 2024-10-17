import { NextResponse } from 'next/server';
import { readdir } from 'fs/promises';
import { join } from 'path';

export async function GET() {
  const uploadDir = join(process.cwd(), 'public', 'uploads');
  
  try {
    const files = await readdir(uploadDir);
    return NextResponse.json({ files });
  } catch (error) {
    console.error('Error reading upload directory:', error);
    return NextResponse.json({ error: 'Error reading upload directory' }, { status: 500 });
  }
}