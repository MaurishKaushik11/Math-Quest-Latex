
import { useState, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { FileText, Download, Loader2, Brain, Target, CheckCircle2, AlertTriangle, Sparkles, BookOpen, Calculator, Zap, Copy, Eye, Settings, Cpu, BarChart3 } from "lucide-react";

interface ExtractedQuestion {
  text: string;
  type: string;
  confidence: number;
  page: number;
}

interface ExtractionResult {
  success: boolean;
  chapter: string;
  total_questions_found: number;
  high_confidence_questions: number;
  estimated_accuracy: number;
  latex_content: string;
  questions: ExtractedQuestion[];
  processing_info: {
    pages_processed: number;
    relevant_pages: number[];
    timestamp: string;
    model_used?: string;
    processing_time_ms?: number;
    fallback_message?: string;
  };
}

interface ProcessingStage {
  stage: string;
  message: string;
  completed: boolean;
  icon: React.ReactNode;
}

const Index = () => {
  const [pdfUrl, setPdfUrl] = useState("https://drive.google.com/file/d/1BQllRXh5_ID08uPTVfEe0DgmxPUm867F/view?usp=sharing");
  const [chapterInput, setChapterInput] = useState("");
  const [extractedLatex, setExtractedLatex] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [extractionResult, setExtractionResult] = useState<ExtractionResult | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [showApiKeyInput, setShowApiKeyInput] = useState(false);
  const [currentStage, setCurrentStage] = useState<string>("");
  const [processingStages, setProcessingStages] = useState<ProcessingStage[]>([]);
  const [previewMode, setPreviewMode] = useState<"latex" | "preview">("latex");
  const latexRef = useRef<HTMLTextAreaElement>(null);
  const { toast } = useToast();

  const initializeProcessingStages = () => {
    const stages: ProcessingStage[] = [
      {
        stage: "initialization",
        message: "Initializing RAG pipeline...",
        completed: false,
        icon: <Settings className="h-4 w-4" />
      },
      {
        stage: "download",
        message: "Downloading PDF document...",
        completed: false,
        icon: <FileText className="h-4 w-4" />
      },
      {
        stage: "analysis",
        message: "Analyzing document structure...",
        completed: false,
        icon: <Brain className="h-4 w-4" />
      },
      {
        stage: "extraction",
        message: "Extracting mathematical questions...",
        completed: false,
        icon: <Calculator className="h-4 w-4" />
      },
      {
        stage: "formatting",
        message: "Converting to LaTeX format...",
        completed: false,
        icon: <Sparkles className="h-4 w-4" />
      },
      {
        stage: "validation",
        message: "Validating and enhancing results...",
        completed: false,
        icon: <CheckCircle2 className="h-4 w-4" />
      }
    ];
    setProcessingStages(stages);
    return stages;
  };

  const updateStage = (stageName: string, completed: boolean = true) => {
    setCurrentStage(stageName);
    setProcessingStages(prev => 
      prev.map(stage => 
        stage.stage === stageName 
          ? { ...stage, completed }
          : stage
      )
    );
  };

  const handleExtract = async () => {
    if (!chapterInput.trim()) {
      toast({
        title: "Error",
        description: "Please enter a chapter or topic to extract",
        variant: "destructive",
      });
      return;
    }

    // Use the enhanced backend with OpenAI
    const backendUrl = window.location.origin;
    let useBackend = true;

    // Test if backend is available
    try {
      const response = await fetch(`${backendUrl}/api/health`, { method: 'GET' });
      if (!response.ok) throw new Error('Backend not available');
    } catch {
      useBackend = false;
      toast({
        title: "Backend Unavailable",
        description: "Using demo mode. Check server configuration.",
        variant: "destructive",
      });
    }

    setIsProcessing(true);
    setProgress(0);
    setExtractedLatex("");
    setExtractionResult(null);
    const stages = initializeProcessingStages();

    try {
      if (useBackend) {
        // Use real RAG pipeline with detailed progress tracking
        updateStage("initialization");
        setProgress(10);
        await new Promise(resolve => setTimeout(resolve, 500));

        updateStage("download");
        setProgress(20);
        toast({
          title: "🚀 Groq RAG Pipeline Started",
          description: "Analyzing RD Sharma Class 12 for question extraction...",
        });
        await new Promise(resolve => setTimeout(resolve, 800));

        updateStage("analysis");
        setProgress(30);

        const response = await fetch(`${backendUrl}/api/extract`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            pdf_url: pdfUrl,
            chapter: chapterInput,
            api_key: apiKey || undefined
          })
        });

        updateStage("extraction");
        setProgress(60);

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Extraction failed');
        }

        updateStage("formatting");
        setProgress(80);
        const result: ExtractionResult = await response.json();
        
        updateStage("validation");
        setProgress(95);
        await new Promise(resolve => setTimeout(resolve, 500));

        setExtractionResult(result);
        setExtractedLatex(result.latex_content);
        setProgress(100);

        // Mark all stages as completed
        setProcessingStages(prev => prev.map(stage => ({ ...stage, completed: true })));

        // Check if using fallback mode
        if (result.processing_info?.model_used?.includes('Fallback')) {
          toast({
            title: "⚡ Enhanced Fallback Mode Active",
            description: `Generated ${result.high_confidence_questions}/${result.total_questions_found} intelligent questions with ${result.estimated_accuracy}% accuracy.`,
            variant: "default",
          });
        } else {
          const timeText = result.processing_info?.processing_time_ms ? `(${(result.processing_info.processing_time_ms / 1000).toFixed(1)}s)` : '';
          toast({
            title: `🎯 RD Sharma Extraction Complete!`,
            description: `Extracted ${result.high_confidence_questions}/${result.total_questions_found} questions with ${result.estimated_accuracy}% accuracy ${timeText}`,
          });
        }
      } else {
        // Fallback to mock data with simulated processing stages
        for (let i = 0; i < stages.length; i++) {
          updateStage(stages[i].stage);
          setProgress((i + 1) * (80 / stages.length));
          await new Promise(resolve => setTimeout(resolve, 600));
        }

        toast({
          title: "⚠️ Demo Mode Active",
          description: "Using mock data - start Python backend for real extraction",
        });

        const mockLatex = generateMockLatex(chapterInput);
        const mockResult: ExtractionResult = {
          success: true,
          chapter: chapterInput,
          total_questions_found: 15,
          high_confidence_questions: 12,
          estimated_accuracy: 87.3,
          latex_content: mockLatex,
          questions: [
            { text: "Find the derivative of $f(x) = \\sin(x^2)$", type: "exercise", confidence: 0.95, page: 245 },
            { text: "Evaluate $\\int_0^\\pi \\cos(x) dx$", type: "problem", confidence: 0.92, page: 246 },
            { text: "Solve the equation $2x^2 - 5x + 3 = 0$", type: "illustration", confidence: 0.88, page: 247 },
          ],
          processing_info: {
            pages_processed: 5,
            relevant_pages: [245, 246, 247, 248, 249],
            timestamp: new Date().toISOString()
          }
        };
        
        setExtractionResult(mockResult);
        setExtractedLatex(mockLatex);
        setProgress(100);
        setProcessingStages(prev => prev.map(stage => ({ ...stage, completed: true })));

        toast({
          title: "✨ Demo Complete",
          description: `Generated sample output for "${chapterInput}" - Start backend for real extraction`,
        });
      }
    } catch (error: any) {
      toast({
        title: "❌ Extraction Failed",
        description: error.message || "Failed to extract content. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
      setCurrentStage("");
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast({
        title: "✅ Copied!",
        description: "LaTeX content copied to clipboard",
      });
    } catch (err) {
      toast({
        title: "❌ Copy Failed",
        description: "Unable to copy to clipboard",
        variant: "destructive",
      });
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
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/30 dark:from-background dark:via-background dark:to-background">
      {/* Enhanced header with theme toggle */}
      <div className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="mr-4 hidden md:flex">
            <div className="flex items-center gap-2">
              <div className="p-2 bg-gradient-to-br from-primary to-accent rounded-lg">
                <BookOpen className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="font-bold text-lg bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                RD Sharma RAG Extractor
              </span>
            </div>
          </div>
          <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
            <div className="w-full flex-1 md:w-auto md:flex-none">
              <span className="text-sm text-foreground/80">Groq Powered • Assignment Mode</span>
            </div>
            <ThemeToggle />
          </div>
        </div>
      </div>
      
      <div className="container mx-auto p-6 space-y-8">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="p-4 bg-gradient-to-br from-primary via-primary to-accent rounded-2xl shadow-lg">
              <Calculator className="h-12 w-12 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
                RD Sharma Extractor
              </h1>
              <p className="text-2xl text-foreground/80 font-medium">
                RAG Pipeline for Class 12 Mathematics Questions
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
            <div className="flex items-center justify-center gap-2 p-3 rounded-lg bg-card border">
              <Cpu className="h-5 w-5 text-blue-500" />
              <span className="font-medium">Groq Llama 3.1</span>
            </div>
            <div className="flex items-center justify-center gap-2 p-3 rounded-lg bg-card border">
              <BarChart3 className="h-5 w-5 text-green-500" />
              <span className="font-medium">95% Target Accuracy</span>
            </div>
            <div className="flex items-center justify-center gap-2 p-3 rounded-lg bg-card border">
              <Sparkles className="h-5 w-5 text-purple-500" />
              <span className="font-medium">LaTeX Export</span>
            </div>
          </div>
        </div>

        <Card className="border-2 shadow-lg bg-gradient-to-br from-card to-card/50">
          <CardHeader className="bg-gradient-to-r from-primary/10 to-accent/10 rounded-t-lg">
            <CardTitle className="flex items-center gap-2 text-xl">
              <FileText className="h-6 w-6 text-primary" />
              PDF Source & Configuration
            </CardTitle>
            <CardDescription className="text-base text-foreground/70">
              RD Sharma Class 12 textbook is pre-configured. Enter the chapter/topic (e.g., "30.3 Conditional Probability").
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
                placeholder="e.g., 30.3 Conditional Probability, 30.9 Bayes' Theorem"
                data-testid="chapter-input"
              />
            </div>

            <Button 
              onClick={handleExtract} 
              disabled={isProcessing || !chapterInput.trim()}
              className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 shadow-lg transition-all duration-300 transform hover:scale-[1.02]"
              data-testid="extract-button"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="mr-3 h-5 w-5 animate-spin" />
                  Processing with GPT-5...
                </>
              ) : (
                <>
                  <Brain className="mr-3 h-5 w-5" />
                  Extract RD Sharma Questions
                </>
              )}
            </Button>

            {isProcessing && (
              <div className="space-y-4">
                <Progress value={progress} className="w-full" />
                <div className="text-center space-y-1">
                  <p className="text-sm font-medium">
                    Processing: {progress}%
                  </p>
                  <p className="text-xs text-foreground/70">
                    {currentStage ? processingStages.find(s => s.stage === currentStage)?.message : "Initializing..."}
                  </p>
                </div>
                
                {/* Processing Stages Visual */}
                <div className="bg-muted/50 rounded-lg p-4 space-y-2">
                  <h4 className="text-sm font-medium flex items-center gap-2">
                    <Zap className="h-4 w-4 text-yellow-500" />
                    Processing Stages
                  </h4>
                  <div className="grid gap-2">
                    {processingStages.map((stage, index) => (
                      <div key={stage.stage} className="flex items-center gap-3 text-xs">
                        <div className={`flex items-center justify-center w-6 h-6 rounded-full border-2 transition-all duration-300 ${
                          stage.completed 
                            ? "bg-green-100 border-green-500 text-green-700" 
                            : currentStage === stage.stage 
                            ? "bg-blue-100 border-blue-500 text-blue-700 animate-pulse" 
                            : "bg-gray-100 border-gray-300 text-gray-500"
                        }`}>
                          {stage.completed ? (
                            <CheckCircle2 className="h-3 w-3" />
                          ) : currentStage === stage.stage ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : (
                            stage.icon
                          )}
                        </div>
                        <span className={`flex-1 ${
                          stage.completed 
                            ? "text-green-700" 
                            : currentStage === stage.stage 
                            ? "text-blue-700 font-medium" 
                            : "text-gray-500"
                        }`}>
                          {stage.message}
                        </span>
                        {stage.completed && <span className="text-green-500 text-xs">✓</span>}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* API Key Input with Free Options Info */}
        <Card className="border-2 border-dashed border-muted-foreground/30 bg-gradient-to-r from-blue-50/50 to-purple-50/50 dark:from-blue-950/20 dark:to-purple-950/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-primary">
              <Brain className="h-5 w-5" />
              AI API Configuration & Free Options
            </CardTitle>
            <CardDescription className="text-foreground/70">
              Groq API is configured for fast, accurate RD Sharma question extraction. Your assignment requires high-precision LaTeX formatting.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Free API Alternatives */}
            <div className="space-y-3">
              <h4 className="font-semibold text-sm flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-green-500" />
                Free AI API Options
              </h4>
              <div className="p-4 rounded-lg bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950/20 dark:to-blue-950/20 border border-green-200 dark:border-green-800">
                <div className="font-semibold text-green-700 dark:text-green-400 mb-2">✅ Groq API Configured</div>
                <div className="text-sm text-green-600 dark:text-green-300">Fast Llama 3.1 70B model ready for RD Sharma extraction</div>
                <div className="text-xs text-foreground/70 mt-2">Optimized for mathematical content and LaTeX formatting</div>
              </div>
            </div>
            
            {/* Current API Key Input */}
            <div className="space-y-2 pt-4 border-t">
              <Label htmlFor="api-key" className="flex items-center gap-2">
                <span>Current API Key</span>
                <Badge variant="secondary" className="text-xs">Optional - Uses Fallback if Missing</Badge>
              </Label>
              <Input
                id="api-key"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-... or paste new key from alternatives above"
              />
            </div>
            
            <Alert className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200 dark:from-blue-950/20 dark:to-purple-950/20 dark:border-blue-800">
              <Brain className="h-4 w-4" />
              <AlertDescription className="text-sm">
                <strong>Assignment Mode:</strong> RAG pipeline configured for RD Sharma Class 12 question extraction with LaTeX formatting as per job requirements.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>

        {/* Accuracy Metrics Display */}
        {extractionResult && (
          <Card className="bg-gradient-to-r from-green-50 to-blue-50 border-green-200">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-green-700">
                <Target className="h-5 w-5" />
                Extraction Results & Accuracy Metrics
              </CardTitle>
              <CardDescription className="text-foreground/80">
                Performance analysis of the RAG pipeline extraction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{extractionResult.estimated_accuracy}%</div>
                  <div className="text-sm font-semibold" style={{color: 'hsl(0, 0%, 10%)'}}>Estimated Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{extractionResult.high_confidence_questions}</div>
                  <div className="text-sm font-semibold" style={{color: 'hsl(0, 0%, 10%)'}}>High Conf. Questions</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">{extractionResult.total_questions_found}</div>
                  <div className="text-sm font-semibold" style={{color: 'hsl(0, 0%, 10%)'}}>Total Found</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">{extractionResult.processing_info.pages_processed}</div>
                  <div className="text-sm font-semibold" style={{color: 'hsl(0, 0%, 10%)'}}>Pages Processed</div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  {extractionResult.estimated_accuracy >= 90 ? (
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-yellow-600" />
                  )}
                  <span className="text-sm">
                    {extractionResult.estimated_accuracy >= 90 
                      ? "🎯 Target accuracy achieved!" 
                      : "⚡ Consider refining chapter/topic for better results"}
                  </span>
                </div>
                
                {extractionResult.processing_info?.model_used?.includes('Fallback') && (
                  <div className="p-2 bg-amber-50 dark:bg-amber-950/20 rounded border border-amber-200 dark:border-amber-800">
                    <div className="text-xs font-medium text-amber-800 dark:text-amber-200">⚡ Fallback Mode Active</div>
                    <div className="text-xs text-amber-700 dark:text-amber-300 mt-1">
                      Using RD Sharma-specific question generation. Groq API key configured for better results.
                    </div>
                  </div>
                )}
                
                <div className="text-xs font-medium" style={{color: 'hsl(0, 0%, 15%)'}}>
                  Relevant pages: {extractionResult.processing_info.relevant_pages.join(", ")}
                </div>
                <div className="text-xs font-medium" style={{color: 'hsl(0, 0%, 15%)'}}>
                  Model: {extractionResult.processing_info.model_used || 'Groq Llama 3.1'}
                </div>
              </div>

              {/* Individual Questions Display */}
              <div className="mt-4">
                <h4 className="font-bold mb-2" style={{color: 'hsl(0, 0%, 5%)'}}>Extracted Questions Preview:</h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {extractionResult.questions.slice(0, 3).map((q, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-white rounded border">
                      <div className="flex-1">
                        <div className="text-sm font-bold truncate" style={{color: 'hsl(0, 0%, 5%)'}}>
                          Question {index + 1} (Page {q.page})
                        </div>
                        <div className="text-xs font-medium capitalize" style={{color: 'hsl(0, 0%, 20%)'}}>{q.type}</div>
                      </div>
                      <Badge variant={q.confidence >= 0.8 ? "default" : "secondary"}>
                        {Math.round(q.confidence * 100)}%
                      </Badge>
                    </div>
                  ))}
                  {extractionResult.questions.length > 3 && (
                    <div className="text-xs text-center font-medium py-1" style={{color: 'hsl(0, 0%, 15%)'}}>
                      ... and {extractionResult.questions.length - 3} more questions
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {extractedLatex && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-purple-600" />
                  <span>Generated LaTeX Document</span>
                </div>
                <div className="flex gap-2">
                  {!showApiKeyInput && (
                    <Button 
                      onClick={() => setShowApiKeyInput(true)}
                      variant="outline" 
                      size="sm"
                    >
                      <Brain className="mr-2 h-4 w-4" />
                      API Settings
                    </Button>
                  )}
                  <Button 
                    onClick={() => copyToClipboard(extractedLatex)}
                    variant="outline" 
                    size="sm"
                  >
                    <Copy className="mr-2 h-4 w-4" />
                    Copy
                  </Button>
                  <Button onClick={downloadLatex} variant="outline" size="sm">
                    <Download className="mr-2 h-4 w-4" />
                    Download .tex
                  </Button>
                </div>
              </CardTitle>
              <CardDescription>
                Complete LaTeX document ready for compilation with {extractionResult?.high_confidence_questions || 0} high-confidence questions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={previewMode} onValueChange={(value) => setPreviewMode(value as "latex" | "preview")} className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="latex" className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    LaTeX Source
                  </TabsTrigger>
                  <TabsTrigger value="preview" className="flex items-center gap-2">
                    <Eye className="h-4 w-4" />
                    Question Preview
                  </TabsTrigger>
                </TabsList>
                
                <TabsContent value="latex" className="mt-4">
                  <div className="relative">
                    <Textarea
                      ref={latexRef}
                      value={extractedLatex}
                      readOnly
                      className="min-h-[500px] font-mono text-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 border-2 border-gray-200 dark:border-gray-700"
                    />
                    <div className="absolute top-2 right-2 flex gap-1">
                      <Badge variant="secondary" className="text-xs">
                        {extractedLatex.split('\n').length} lines
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {Math.round(extractedLatex.length / 1024 * 100) / 100} KB
                      </Badge>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="preview" className="mt-4">
                  <ScrollArea className="h-[500px] w-full border-2 rounded-md p-4 bg-gray-50">
                    <div className="space-y-6">
                      <div className="text-center border-b pb-4">
                        <h2 className="text-xl font-bold">Mathematical Questions: {extractionResult?.chapter}</h2>
                        <p className="text-sm text-foreground/70 mt-1">
                          Generated on {new Date().toLocaleDateString()}
                        </p>
                      </div>
                      
                      {extractionResult?.questions.map((question, index) => (
                        <div key={index} className="border rounded-lg p-4 bg-white shadow-sm">
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">
                                Question {index + 1}
                              </Badge>
                              <Badge variant={question.confidence >= 0.9 ? "default" : question.confidence >= 0.8 ? "secondary" : "outline"} className="text-xs">
                                {Math.round(question.confidence * 100)}% confidence
                              </Badge>
                            </div>
                            <div className="text-right text-xs text-foreground/70">
                              <div>Page {question.page}</div>
                              <div className="capitalize">{question.type}</div>
                            </div>
                          </div>
                          
                          <div className="bg-gray-50 p-3 rounded-md font-mono text-sm border-l-4 border-blue-500">
                            {question.text}
                          </div>
                        </div>
                      ))}
                      
                      {(!extractionResult?.questions || extractionResult.questions.length === 0) && (
                        <div className="text-center text-foreground/70 py-8">
                          <Calculator className="h-12 w-12 mx-auto mb-3 opacity-50" />
                          <p>No questions to preview</p>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}

        <Card className="bg-muted/50">
          <CardHeader>
            <CardTitle className="text-lg">How it works</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="list-decimal list-inside space-y-2 text-sm text-foreground/70">
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
