import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionAI · Professional CIFAR-10 Neural Classifier",
  description:
    "Production-grade image classification web application utilizing Transfer Learning with MobileNetV2 fine-tuned on the CIFAR-10 dataset.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased selection:bg-indigo-500/30 selection:text-indigo-200">
        {children}
      </body>
    </html>
  );
}

