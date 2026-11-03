import { NextRequest, NextResponse } from "next/server";
import { execFile } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";
import path from "path";
import os from "os";
import { PredictionResponse } from "@/lib/types";

const execFileAsync = promisify(execFile);

export async function POST(req: NextRequest) {
  let tempFilePath: string | null = null;

  try {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return NextResponse.json(
        { success: false, error: "No image file provided in request." },
        { status: 400 }
      );
    }

    // Convert file to buffer and write to temp directory
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const tempDir = os.tmpdir();
    const fileName = `vision_${Date.now()}_${Math.random().toString(36).slice(2)}.png`;
    tempFilePath = path.join(tempDir, fileName);

    await fs.writeFile(tempFilePath, buffer);

    // Path to the Python virtual environment and bridge script
    // Finds predict_api.py whether invoked from frontend/, moduleD_image_classifier/, or root
    const cwd = process.cwd();
    const candidateScriptPaths = [
      path.resolve(cwd, "..", "predict_api.py"),               // When run from frontend/
      path.resolve(cwd, "predict_api.py"),                    // When run from moduleD_image_classifier/
      path.resolve(cwd, "Modules", "moduleD_image_classifier", "predict_api.py"), // When run from root
    ];

    let scriptPath = candidateScriptPaths[0];
    for (const p of candidateScriptPaths) {
      try {
        await fs.access(p);
        scriptPath = p;
        break;
      } catch {}
    }

    const moduleDir = path.dirname(scriptPath);
    const repoRoot = path.resolve(moduleDir, "..", "..");
    const venvPython = path.join(repoRoot, ".venv", "Scripts", "python.exe");

    // Check if the venv python exists, otherwise fallback to system python
    let pythonExecutable = venvPython;
    try {
      await fs.access(venvPython);
    } catch {
      pythonExecutable = "python";
    }

    // Execute Python prediction bridge
    const { stdout, stderr } = await execFileAsync(pythonExecutable, [
      scriptPath,
      tempFilePath,
    ]);

    if (stderr && !stdout) {
      console.error("Python bridge stderr:", stderr);
      return NextResponse.json(
        { success: false, error: stderr.trim() },
        { status: 500 }
      );
    }

    const result = JSON.parse(stdout.trim()) as PredictionResponse;
    return NextResponse.json(result);
  } catch (err: any) {
    console.error("Prediction API error:", err);
    return NextResponse.json(
      {
        success: false,
        error: err.message || "An unexpected error occurred during prediction.",
      },
      { status: 500 }
    );
  } finally {
    // Clean up temporary image file
    if (tempFilePath) {
      try {
        await fs.unlink(tempFilePath);
      } catch (cleanupErr) {
        // Ignore file cleanup error
      }
    }
  }
}

