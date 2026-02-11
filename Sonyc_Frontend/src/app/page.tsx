"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Youtube, FileText, Github, Globe, Sparkles, Zap, Shield, Users } from "lucide-react";
import { useRouter } from "next/navigation";
import { MotionBackground } from "@/components/motion-background";
import { ThemeSwitcher } from "@/components/theme-switcher";
import Image from "next/image";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
}

function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const mouseRef = useRef({ x: 0, y: 0 });
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const getPrimaryColor = () => {
      const style = getComputedStyle(document.documentElement);
      const hsl = style.getPropertyValue("--primary").trim();
      return hsl;
    };

    const hslToRgb = (hslString: string) => {
      const values = hslString.split(" ").map((v) => parseFloat(v));
      const h = values[0] / 360;
      const s = values[1] / 100;
      const l = values[2] / 100;

      let r, g, b;

      if (s === 0) {
        r = g = b = l;
      } else {
        const hue2rgb = (p: number, q: number, t: number) => {
          if (t < 0) t += 1;
          if (t > 1) t -= 1;
          if (t < 1 / 6) return p + (q - p) * 6 * t;
          if (t < 1 / 2) return q;
          if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
          return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
      }

      return {
        r: Math.round(r * 255),
        g: Math.round(g * 255),
        b: Math.round(b * 255),
      };
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };

      for (let i = 0; i < 3; i++) {
        particlesRef.current.push({
          x: e.clientX,
          y: e.clientY,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          life: 100,
          maxLife: 100,
        });
      }

      if (particlesRef.current.length > 150) {
        particlesRef.current = particlesRef.current.slice(-150);
      }
    };

    window.addEventListener("mousemove", handleMouseMove);

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const primaryHsl = getPrimaryColor();
      const primaryRgb = hslToRgb(primaryHsl);

      particlesRef.current = particlesRef.current.filter((particle) => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.life -= 1;

        if (particle.life <= 0) return false;

        const alpha = particle.life / particle.maxLife;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${primaryRgb.r}, ${primaryRgb.g}, ${primaryRgb.b}, ${alpha * 0.6})`;
        ctx.fill();

        return true;
      });

      for (let i = 0; i < particlesRef.current.length; i++) {
        for (let j = i + 1; j < particlesRef.current.length; j++) {
          const dx = particlesRef.current[i].x - particlesRef.current[j].x;
          const dy = particlesRef.current[i].y - particlesRef.current[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100) {
            const alpha =
              (1 - distance / 100) *
              (particlesRef.current[i].life / particlesRef.current[i].maxLife) *
              (particlesRef.current[j].life / particlesRef.current[j].maxLife);
            ctx.beginPath();
            ctx.moveTo(particlesRef.current[i].x, particlesRef.current[i].y);
            ctx.lineTo(particlesRef.current[j].x, particlesRef.current[j].y);
            ctx.strokeStyle = `rgba(${primaryRgb.r}, ${primaryRgb.g}, ${primaryRgb.b}, ${alpha * 0.3})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      window.removeEventListener("mousemove", handleMouseMove);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none"
      style={{ opacity: 0.8 }}
    />
  );
}

export default function HomePage() {
  const [isSignupOpen, setSignupOpen] = useState(false);
  const [isSigninOpen, setSigninOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();
  const { toast } = useToast();

  // Check if user is already authenticated
  useEffect(() => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    if (token) {
      // Verify token is still valid
      api.getCurrentUser()
        .then(() => {
          // Token is valid, redirect to chat
          router.push('/chat');
        })
        .catch(() => {
          // Token is invalid, clear it
          api.logout();
        });
    }
  }, [router]);

  const handleAuthSuccess = () => {
    router.push("/chat");
  };

  const handleSwitchToSignup = () => {
    setSigninOpen(false);
    setSignupOpen(true);
    setError("");
  };

  const handleSwitchToSignin = () => {
    setSignupOpen(false);
    setSigninOpen(true);
    setError("");
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);
    
    try {
      await api.signup(email, password);
      toast({
        title: "Account created!",
        description: "Welcome to SONYC!",
      });
      handleAuthSuccess();
    } catch (err: any) {
      setError(err.message || "Failed to create account");
      toast({
        title: "Error",
        description: err.message || "Failed to create account",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);
    
    try {
      await api.signin(email, password);
      toast({
        title: "Welcome back!",
        description: "You've been signed in successfully.",
      });
      handleAuthSuccess();
    } catch (err: any) {
      setError(err.message || "Failed to sign in");
      toast({
        title: "Error",
        description: err.message || "Failed to sign in",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const features = [
    {
      icon: <Youtube className="h-8 w-8 text-accent" />,
      title: "YouTube Intelligence",
      description: "Transform any YouTube video into an interactive knowledge base. Ask questions, get summaries, and extract key insights without watching hours of content.",
    },
    {
      icon: <FileText className="h-8 w-8 text-accent" />,
      title: "PDF Analysis",
      description: "Upload documents and instantly chat with your PDFs. Extract information, summarize sections, and find answers buried in lengthy documents.",
    },
    {
      icon: <Github className="h-8 w-8 text-accent" />,
      title: "Code Understanding",
      description: "Decode any GitHub repository with AI-powered explanations. Understand complex codebases, get function explanations, and learn faster.",
    },
    {
      icon: <Globe className="h-8 w-8 text-accent" />,
      title: "Web Scraping",
      description: "Turn any website into a conversational interface. Extract data, get summaries, and interact with web content in natural language.",
    },
  ];

  const benefits = [
    {
      icon: <Sparkles className="h-6 w-6 text-primary" />,
      title: "AI-Powered RAG",
      description: "Advanced Retrieval-Augmented Generation ensures accurate, context-aware responses from your sources.",
    },
    {
      icon: <Zap className="h-6 w-6 text-primary" />,
      title: "Lightning Fast",
      description: "Get instant answers and insights without wasting time searching through content manually.",
    },
    {
      icon: <Shield className="h-6 w-6 text-primary" />,
      title: "Secure & Private",
      description: "Your data is encrypted and processed securely. We prioritize your privacy and data protection.",
    },
    {
      icon: <Users className="h-6 w-6 text-primary" />,
      title: "Built for Everyone",
      description: "Whether you're a student, developer, or professional, SONYC adapts to your workflow.",
    },
  ];

  return (
    <>
      <MotionBackground />
      <ParticleBackground />

      <div className="flex flex-col min-h-[100dvh] home-gradient relative z-1">
        {/* Header */}
        <header className="px-4 lg:px-6 h-16 flex items-center fixed top-4 left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] max-w-7xl z-10 bg-primary/10 backdrop-blur-lg rounded-xl border border-primary/20 shadow-lg">
          <Link href="#" className="flex items-center justify-center" prefetch={false}>
            <Image src="/icon.png" alt="SONYC Logo" width={32} height={32} />
            <span className="ml-2 text-xl font-headline uppercase">Sonyc</span>
          </Link>
          <nav className="ml-auto flex gap-2 sm:gap-4 items-center">
            <ThemeSwitcher />
            <Button variant="ghost" onClick={() => setSigninOpen(true)} className="hidden sm:flex">
              Sign In
            </Button>
            <Button onClick={() => setSignupOpen(true)}>Sign Up</Button>
          </nav>
        </header>

        <main className="flex-1 w-full relative z-1">
          {/* Hero Section */}
          <section className="w-full min-h-screen flex items-center py-20 px-4">
            <div className="container max-w-7xl mx-auto">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
                {/* Left Side - Content */}
                <div className="flex flex-col items-start justify-center space-y-8 text-left">
                  <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 backdrop-blur-sm">
                    <Sparkles className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium">Powered by Advanced RAG Technology</span>
                  </div>

                  <h1 className="font-headline uppercase max-w-5xl font-bold
                               text-[clamp(1.5rem,5vw,2rem)] 
                               sm:text-[clamp(1.75rem,5vw,2.25rem)] 
                               md:text-[clamp(2rem,4vw,2.5rem)] 
                               lg:text-[clamp(2.25rem,3.5vw,3rem)] 
                               xl:text-[clamp(2.5rem,3vw,3.5rem)] 
                               leading-tight">
                    Your Intelligent AI Assistant for Everything
                  </h1>

                  <p className="max-w-3xl text-muted-foreground leading-relaxed 
                              text-[clamp(0.75rem,1.8vw,0.9rem)] 
                              sm:text-[clamp(0.875rem,2vw,1rem)] 
                              md:text-[clamp(0.9rem,2.2vw,1.05rem)] 
                              lg:text-[clamp(1rem,2.4vw,1.1rem)] 
                              xl:text-[clamp(1.05rem,2.6vw,1.15rem)]">
                    <span className="font-bold">SONYC</span> transforms how you interact with digital content. Chat with YouTube videos, PDFs, GitHub repositories, and websites using the power of AI. Get instant answers, summaries, and insights from any source.
                  </p>

                  <div className="flex flex-col sm:flex-row gap-4 pt-6 w-full sm:w-auto">
                    <Button
                      size="lg"
                      onClick={() => setSignupOpen(true)}
                      className="bg-primary hover:bg-primary/90 text-primary-foreground w-full sm:w-auto text-lg h-14 px-8"
                    >
                      Get Started Free
                    </Button>
                    <Button 
                      size="lg" 
                      variant="outline" 
                      className="w-full sm:w-auto text-lg h-14 px-8"
                      onClick={() => {
                        document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
                      }}
                    >
                      Explore Features
                    </Button>
                  </div>

                  <div className="flex flex-wrap items-center gap-8 pt-8 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-green-500"></div>
                      <span>No Credit Card Required</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-green-500"></div>
                      <span>Free to Get Started</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-green-500"></div>
                      <span>Cancel Anytime</span>
                    </div>
                  </div>
                </div>

                {/* Right Side - Logo */}
                <div className="flex items-center justify-center lg:justify-end h-full min-h-[400px] lg:min-h-[600px]">
                  <div className="relative w-full h-full max-w-lg lg:max-w-2xl">
                    <Image
                      src="/logo.png"
                      alt="SONYC Logo"
                      fill
                      className="object-contain"
                      priority
                      sizes="(max-width: 1024px) 100vw, 50vw"
                    />
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Features Section */}
          <section id="features" className="w-full py-20 px-4">
            <div className="container max-w-7xl mx-auto">
              <div className="flex flex-col items-center justify-center space-y-6 text-center mb-16">
                <div className="inline-block rounded-lg bg-muted px-4 py-2 text-sm font-medium">
                  Core Features
                </div>
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl max-w-3xl">
                  One Platform, Infinite Possibilities
                </h2>
                <p className="max-w-3xl text-muted-foreground text-base sm:text-lg md:text-xl leading-relaxed">
                  SONYC seamlessly integrates with your favorite platforms to deliver intelligent, context-aware assistance across all your digital content.
                </p>
              </div>
              
              <div className="grid gap-8 sm:grid-cols-1 md:grid-cols-2 max-w-6xl mx-auto">
                {features.map((feature, index) => (
                  <div 
                    key={index} 
                    className="group flex flex-col sm:flex-row gap-6 p-8 rounded-2xl bg-primary/5 backdrop-blur-lg border border-primary/10 shadow-lg transition-all hover:scale-105 hover:bg-primary/10 hover:border-primary/30 hover:shadow-xl"
                  >
                    <div className="flex-shrink-0 mx-auto sm:mx-0">
                      <div className="p-4 rounded-xl bg-primary/10 group-hover:bg-primary/20 transition-colors">
                        {feature.icon}
                      </div>
                    </div>
                    <div className="flex flex-col gap-3 text-center sm:text-left">
                      <h3 className="text-xl sm:text-2xl font-bold">{feature.title}</h3>
                      <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Benefits Section */}
          <section className="w-full py-20 px-4 bg-primary/5">
            <div className="container max-w-7xl mx-auto">
              <div className="flex flex-col items-center justify-center space-y-6 text-center mb-16">
                <div className="inline-block rounded-lg bg-muted px-4 py-2 text-sm font-medium">
                  Why Choose SONYC
                </div>
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl max-w-3xl">
                  Experience the Future of Content Interaction
                </h2>
                <p className="max-w-3xl text-muted-foreground text-base sm:text-lg md:text-xl leading-relaxed">
                  Built with cutting-edge AI technology to revolutionize how you consume and understand digital content.
                </p>
              </div>
              
              <div className="grid gap-8 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-4 max-w-7xl mx-auto">
                {benefits.map((benefit, index) => (
                  <div 
                    key={index} 
                    className="flex flex-col items-center text-center gap-4 p-6 rounded-xl bg-background/50 backdrop-blur-sm border border-primary/10 shadow-md hover:shadow-lg transition-all"
                  >
                    <div className="p-3 rounded-full bg-primary/10">
                      {benefit.icon}
                    </div>
                    <h3 className="text-lg font-bold">{benefit.title}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {benefit.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* CTA Section */}
          <section className="w-full py-20 px-4">
            <div className="container max-w-5xl mx-auto">
              <div className="flex flex-col items-center justify-center space-y-8 text-center p-12 rounded-3xl bg-gradient-to-br from-primary/20 to-accent/20 border border-primary/30 backdrop-blur-lg shadow-2xl">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl max-w-3xl">
                  Ready to Transform Your Workflow?
                </h2>
                <p className="max-w-2xl text-muted-foreground text-base sm:text-lg md:text-xl leading-relaxed">
                  Join thousands of users who are already saving hours every day with SONYC's intelligent AI assistance.
                </p>
                <Button
                  size="lg"
                  onClick={() => setSignupOpen(true)}
                  className="bg-primary hover:bg-primary/90 text-primary-foreground text-lg h-14 px-10"
                >
                  Start Your Free Trial
                </Button>
              </div>
            </div>
          </section>
        </main>

        {/* Footer */}
        <footer className="w-full py-8 px-4 border-t relative z-1">
          <div className="container max-w-7xl mx-auto">
            <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
              <p className="text-sm text-muted-foreground text-center sm:text-left">
                &copy; {new Date().getFullYear()} SONYC. All rights reserved.
              </p>
              <nav className="flex gap-6 justify-center">
                <Link href="#" className="text-sm hover:underline underline-offset-4 text-muted-foreground hover:text-foreground transition-colors" prefetch={false}>
                  Terms of Service
                </Link>
                <Link href="#" className="text-sm hover:underline underline-offset-4 text-muted-foreground hover:text-foreground transition-colors" prefetch={false}>
                  Privacy Policy
                </Link>
                <Link href="#" className="text-sm hover:underline underline-offset-4 text-muted-foreground hover:text-foreground transition-colors" prefetch={false}>
                  Contact Us
                </Link>
              </nav>
            </div>
          </div>
        </footer>
      </div>

      {/* Sign Up Dialog */}
      <Dialog open={isSignupOpen} onOpenChange={setSignupOpen}>
        <DialogContent className="sm:max-w-md bg-background/95 backdrop-blur-xl border-primary/20">
          <DialogHeader className="text-center space-y-2">
            <DialogTitle className="text-2xl font-bold">Create Your Account</DialogTitle>
            <DialogDescription className="text-base">
              Join SONYC and unlock the power of AI-driven content interaction.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <Button variant="outline" className="w-full h-11 text-base">
              <svg role="img" viewBox="0 0 24 24" className="mr-2 h-5 w-5"><path fill="currentColor" d="M12.48 10.92v3.28h7.84c-.24 1.84-.85 3.18-1.73 4.1-1.02 1.02-2.3 1.62-3.96 1.62-3.33 0-6.03-2.7-6.03-6.02s2.7-6.02 6.03-6.02c1.87 0 3.14.75 3.88 1.48l2.54-2.54C16.83 2.66 14.88 2 12.48 2 7.33 2 3.16 6.15 3.16 11.3s4.17 9.3 9.32 9.3c2.6 0 4.5-1.02 6.13-2.66 1.7-1.7 2.36-4.02 2.36-6.52v-.5H12.48z"></path></svg>
              Continue with Google
            </Button>
            <Button variant="outline" className="w-full h-11 text-base">
              <Github className="mr-2 h-5 w-5" />
              Continue with GitHub
            </Button>
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">
                  Or continue with email
                </span>
              </div>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="name" className="text-sm font-medium">Full Name</Label>
              <Input id="name" placeholder="John Doe" className="h-11" />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="email-signup" className="text-sm font-medium">Email Address</Label>
              <Input 
                id="email-signup" 
                type="email" 
                placeholder="john@example.com" 
                className="h-11"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="password-signup" className="text-sm font-medium">Password</Label>
              <Input 
                id="password-signup" 
                type="password" 
                placeholder="••••••••" 
                className="h-11"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            {error && (
              <div className="text-sm text-destructive">{error}</div>
            )}
          </div>
          <DialogFooter className="flex-col gap-3">
            <form onSubmit={handleSignup} className="w-full">
              <Button type="submit" disabled={isLoading} className="w-full h-11 text-base">
                {isLoading ? "Creating..." : "Create Account"}
            </Button>
            </form>
            <p className="text-center text-sm text-muted-foreground">
              Already have an account?{' '}
              <button onClick={handleSwitchToSignin} className="underline font-semibold hover:text-primary transition-colors">
                Sign In
              </button>
            </p>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Sign In Dialog */}
      <Dialog open={isSigninOpen} onOpenChange={setSigninOpen}>
        <DialogContent className="sm:max-w-md bg-background/95 backdrop-blur-xl border-primary/20">
          <DialogHeader className="text-center space-y-2">
            <DialogTitle className="text-2xl font-bold">Welcome Back</DialogTitle>
            <DialogDescription className="text-base">
              Sign in to continue your SONYC experience.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <Button variant="outline" className="w-full h-11 text-base">
              <svg role="img" viewBox="0 0 24 24" className="mr-2 h-5 w-5"><path fill="currentColor" d="M12.48 10.92v3.28h7.84c-.24 1.84-.85 3.18-1.73 4.1-1.02 1.02-2.3 1.62-3.96 1.62-3.33 0-6.03-2.7-6.03-6.02s2.7-6.02 6.03-6.02c1.87 0 3.14.75 3.88 1.48l2.54-2.54C16.83 2.66 14.88 2 12.48 2 7.33 2 3.16 6.15 3.16 11.3s4.17 9.3 9.32 9.3c2.6 0 4.5-1.02 6.13-2.66 1.7-1.7 2.36-4.02 2.36-6.52v-.5H12.48z"></path></svg>
              Continue with Google
            </Button>
            <Button variant="outline" className="w-full h-11 text-base">
              <Github className="mr-2 h-5 w-5" />
              Continue with GitHub
            </Button>
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">
                  Or continue with email
                </span>
              </div>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="email-signin" className="text-sm font-medium">Email Address</Label>
              <Input 
                id="email-signin" 
                type="email" 
                placeholder="john@example.com" 
                className="h-11"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="password-signin" className="text-sm font-medium">Password</Label>
              <Input 
                id="password-signin" 
                type="password" 
                placeholder="••••••••" 
                className="h-11"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            {error && (
              <div className="text-sm text-destructive">{error}</div>
            )}
            <div className="flex items-center justify-between text-sm">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="rounded" />
                <span className="text-muted-foreground">Remember me</span>
              </label>
              <Link href="#" className="text-primary hover:underline">
                Forgot password?
              </Link>
            </div>
          </div>
          <DialogFooter className="flex-col gap-3">
            <form onSubmit={handleSignin} className="w-full">
              <Button type="submit" disabled={isLoading} className="w-full h-11 text-base">
                {isLoading ? "Signing in..." : "Sign In"}
            </Button>
            </form>
            <p className="text-center text-sm text-muted-foreground">
              Don't have an account?{' '}
              <button onClick={handleSwitchToSignup} className="underline font-semibold hover:text-primary transition-colors">
                Sign Up
              </button>
            </p>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}