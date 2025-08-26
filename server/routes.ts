import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { pdfExtractor } from "./openai-service";
import { z } from "zod";

const extractRequestSchema = z.object({
  pdf_url: z.string().url(),
  chapter: z.string().min(1),
  api_key: z.string().optional()
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Health check endpoint
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });

  // Enhanced PDF extraction endpoint
  app.post("/api/extract", async (req, res) => {
    try {
      const validatedData = extractRequestSchema.parse(req.body);
      
      // Override API key if provided in request
      if (validatedData.api_key) {
        process.env.OPENAI_API_KEY = validatedData.api_key;
      }
      
      if (!process.env.OPENAI_API_KEY) {
        return res.status(400).json({ 
          error: "OpenAI API key not configured. Please provide api_key in request or set OPENAI_API_KEY environment variable." 
        });
      }
      
      const result = await pdfExtractor.extractQuestionsFromPDF(
        validatedData.pdf_url,
        validatedData.chapter
      );
      
      // Add helpful message if using fallback mode
      if (result.processing_info?.model_used?.includes('Fallback')) {
        result.processing_info.fallback_message = "API quota exceeded. Using enhanced fallback mode with intelligent question generation.";
      }
      
      res.json(result);
    } catch (error: any) {
      console.error("Extraction error:", error);
      res.status(500).json({ 
        error: error.message || "Failed to extract questions from PDF" 
      });
    }
  });

  // Get extraction history (if we want to store results)
  app.get("/api/extractions", (req, res) => {
    // This could be expanded to return stored extraction history
    res.json({ extractions: [] });
  });

  const httpServer = createServer(app);
  return httpServer;
}
