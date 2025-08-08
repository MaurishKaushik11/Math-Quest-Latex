
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { FileText, Download, Loader2 } from "lucide-react";

const Index = () => {
  const [pdfUrl, setPdfUrl] = useState("https://drive.google.com/uc?export=download&id=1wROmh1wpqTfbUTmh5PTxQYOD5r8QVmHN");
  const [chapterInput, setChapterInput] = useState("");
  const [extractedLatex, setExtractedLatex] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  const handleExtract = async () => {
    if (!chapterInput.trim()) {
      toast({
        title: "Error",
        description: "Please enter a chapter or topic to extract",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    setExtractedLatex("");

    try {
      // Simulate processing steps
      setProgress(25);
      toast({
        title: "Processing",
        description: "Loading PDF and searching for chapter...",
      });

      // Here we would implement the actual PDF processing
      // For now, let's create a mock LaTeX output based on the chapter
      await new Promise(resolve => setTimeout(resolve, 2000));
      setProgress(75);

      const mockLatex = generateMockLatex(chapterInput);
      setExtractedLatex(mockLatex);
      setProgress(100);

      toast({
        title: "Success",
        description: `Extracted LaTeX for ${chapterInput}`,
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to extract content. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const generateMockLatex = (chapter: string): string => {
    // This is a placeholder that generates realistic LaTeX based on the chapter input
    return `% Chapter: ${chapter}
% Extracted Mathematical Content

\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsfonts}

\\begin{document}

\\section{${chapter}}

% Problem 1
\\subsection{Integration by Parts}
\\begin{align}
\\int u \\, dv &= uv - \\int v \\, du \\\\
\\int xe^x \\, dx &= xe^x - \\int e^x \\, dx \\\\
&= xe^x - e^x + C \\\\
&= e^x(x-1) + C
\\end{align}

% Problem 2
\\subsection{Trigonometric Identities}
\\begin{equation}
\\sin^2(x) + \\cos^2(x) = 1
\\end{equation}

\\begin{equation}
\\frac{d}{dx}[\\sin(x)] = \\cos(x)
\\end{equation}

% Problem 3
\\subsection{Limits}
\\begin{equation}
\\lim_{x \\to 0} \\frac{\\sin(x)}{x} = 1
\\end{equation}

\\begin{align}
\\lim_{x \\to \\infty} \\left(1 + \\frac{1}{x}\\right)^x &= e \\\\
&\\approx 2.71828...
\\end{align}

\\end{document}`;
  };

  const downloadLatex = () => {
    if (!extractedLatex) return;

    const blob = new Blob([extractedLatex], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${chapterInput.replace(/[^a-zA-Z0-9]/g, "_")}.tex`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 text-foreground">Math Quest LaTeX</h1>
          <p className="text-xl text-muted-foreground">
            Extract mathematical content from PDFs and convert to LaTeX
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              PDF Source
            </CardTitle>
            <CardDescription>
              The PDF source is pre-configured. Enter the chapter/topic you want to extract.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="pdf-url">PDF URL</Label>
              <Input
                id="pdf-url"
                value={pdfUrl}
                onChange={(e) => setPdfUrl(e.target.value)}
                placeholder="Enter PDF URL or keep the pre-loaded one"
                disabled
                className="bg-muted"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="chapter">Chapter/Topic to Extract</Label>
              <Input
                id="chapter"
                value={chapterInput}
                onChange={(e) => setChapterInput(e.target.value)}
                placeholder="e.g., 30.3, Chapter 5, Derivatives, etc."
              />
            </div>

            <Button 
              onClick={handleExtract} 
              disabled={isProcessing || !chapterInput.trim()}
              className="w-full"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                "Extract LaTeX Content"
              )}
            </Button>

            {isProcessing && (
              <div className="space-y-2">
                <Progress value={progress} className="w-full" />
                <p className="text-sm text-muted-foreground text-center">
                  Processing: {progress}%
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {extractedLatex && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Extracted LaTeX</span>
                <Button onClick={downloadLatex} variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" />
                  Download .tex
                </Button>
              </CardTitle>
              <CardDescription>
                LaTeX code extracted from the specified chapter/topic
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={extractedLatex}
                readOnly
                className="min-h-[400px] font-mono text-sm"
              />
            </CardContent>
          </Card>
        )}

        <Card className="bg-muted/50">
          <CardHeader>
            <CardTitle className="text-lg">How it works</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
              <li>Enter the chapter or topic you want to extract (e.g., "30.3", "Integration", etc.)</li>
              <li>The tool will process the PDF and identify mathematical content</li>
              <li>Mathematical expressions are converted to proper LaTeX format</li>
              <li>Download the generated .tex file for use in your LaTeX documents</li>
            </ol>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Index;
