declare module 'react-pdf' {
  import * as React from 'react';

  export const pdfjs: any;

  export interface DocumentProps {
    file: string | File | { url: string } | any;
    onLoadSuccess?: (info: { numPages: number }) => void;
    loading?: React.ReactNode;
    children?: React.ReactNode;
  }
  export function Document(props: DocumentProps): JSX.Element;

  export interface PageProps {
    pageNumber: number;
    width?: number;
    renderTextLayer?: boolean;
    renderAnnotationLayer?: boolean;
    onRenderSuccess?: () => void;
    customTextRenderer?: (layer: { str: string }) => React.ReactNode;
  }
  export function Page(props: PageProps): JSX.Element;
}


