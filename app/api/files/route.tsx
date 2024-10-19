import { NextResponse } from 'next/server';
import { readdir } from 'fs/promises';
import { join } from 'path';
import { unlink } from 'fs/promises';
import { NextRequest } from 'next/server';

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

export async function DELETE(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const fileName = searchParams.get('file');

  if (!fileName) {
    return NextResponse.json({ error: 'File name is required' }, { status: 400 });
  }

  const filePath = join(process.cwd(), 'public', 'uploads', fileName);

  try {
    await unlink(filePath);
    return NextResponse.json({ message: 'File deleted successfully' });
  } catch (error) {
    console.error('Error deleting file:', error);
    return NextResponse.json({ error: 'Error deleting file' }, { status: 500 });
  }
}