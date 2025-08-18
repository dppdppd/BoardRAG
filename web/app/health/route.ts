import { NextResponse } from 'next/server';

export function GET() {
	return new NextResponse('ok', {
		status: 200,
		headers: { 'content-type': 'text/plain', 'cache-control': 'no-store, no-cache, must-revalidate' },
	});
}

export function HEAD() {
	return new NextResponse(null, { status: 200 });
}


