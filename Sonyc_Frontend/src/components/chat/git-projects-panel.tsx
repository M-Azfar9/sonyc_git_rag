"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter,
} from "@/components/ui/dialog";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import {
    GitBranch,
    RefreshCw,
    Trash2,
    Plus,
    FileText,
    ExternalLink,
    Clock,
    Loader2,
} from "lucide-react";
import type { GitProject } from "@/lib/types";

interface GitProjectsPanelProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function GitProjectsPanel({
    open,
    onOpenChange,
}: GitProjectsPanelProps) {
    const [projects, setProjects] = useState<GitProject[]>([]);
    const [loading, setLoading] = useState(false);
    const [syncing, setSyncing] = useState<number | null>(null);
    const [deleting, setDeleting] = useState<number | null>(null);
    const [showAddForm, setShowAddForm] = useState(false);
    const [newRepoUrl, setNewRepoUrl] = useState("");
    const [newBranch, setNewBranch] = useState("main");
    const [adding, setAdding] = useState(false);
    const [reportLoading, setReportLoading] = useState<number | null>(null);
    const [reportContent, setReportContent] = useState<string | null>(null);
    const [reportType, setReportType] = useState("full");
    const { toast } = useToast();

    const fetchProjects = async () => {
        setLoading(true);
        try {
            const data = await api.getProjects();
            setProjects(data);
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to load projects",
                variant: "destructive",
            });
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (open) {
            fetchProjects();
        }
    }, [open]);

    const handleAddProject = async () => {
        if (!newRepoUrl.trim()) return;
        setAdding(true);
        try {
            await api.createProject({
                repo_url: newRepoUrl,
                branch: newBranch || "main",
            });
            toast({
                title: "Project Added",
                description: "Repository has been ingested successfully",
            });
            setNewRepoUrl("");
            setNewBranch("main");
            setShowAddForm(false);
            fetchProjects();
        } catch (error: any) {
            toast({
                title: "Error",
                description: error.message || "Failed to add project",
                variant: "destructive",
            });
        } finally {
            setAdding(false);
        }
    };

    const handleSync = async (projectId: number) => {
        setSyncing(projectId);
        try {
            await api.syncProject(projectId);
            toast({
                title: "Sync Started",
                description: "Project is being re-ingested in the background",
            });
            // Refresh after a short delay to show updated sync time
            setTimeout(fetchProjects, 2000);
        } catch (error: any) {
            toast({
                title: "Sync Failed",
                description: error.message || "Failed to sync project",
                variant: "destructive",
            });
        } finally {
            setSyncing(null);
        }
    };

    const handleDelete = async (projectId: number) => {
        setDeleting(projectId);
        try {
            await api.deleteProject(projectId);
            toast({ title: "Project Deleted" });
            fetchProjects();
        } catch (error: any) {
            toast({
                title: "Error",
                description: error.message || "Failed to delete project",
                variant: "destructive",
            });
        } finally {
            setDeleting(null);
        }
    };

    const handleGenerateReport = async (projectId: number) => {
        setReportLoading(projectId);
        setReportContent(null);
        try {
            const result = await api.generateGitReport(projectId, reportType);
            setReportContent(result.report);
            toast({ title: "Report Generated" });
        } catch (error: any) {
            toast({
                title: "Error",
                description: error.message || "Failed to generate report",
                variant: "destructive",
            });
        } finally {
            setReportLoading(null);
        }
    };

    const formatSyncTime = (isoString?: string) => {
        if (!isoString) return "Never";
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        if (diffMins < 1) return "Just now";
        if (diffMins < 60) return `${diffMins}m ago`;
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        return date.toLocaleDateString();
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-2xl max-h-[80vh] overflow-y-auto bg-background/95 backdrop-blur-xl">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <GitBranch className="h-5 w-5 text-primary" />
                        Git Projects
                    </DialogTitle>
                    <DialogDescription>
                        Manage your ingested GitHub repositories. Projects are automatically
                        re-synced when pushes are detected via webhooks.
                    </DialogDescription>
                </DialogHeader>

                <div className="space-y-4 py-2">
                    {/* Add Project Form */}
                    {showAddForm ? (
                        <div className="border rounded-lg p-4 space-y-3 bg-muted/30">
                            <div className="space-y-2">
                                <Label htmlFor="repo-url" className="text-sm font-medium">
                                    Repository URL
                                </Label>
                                <Input
                                    id="repo-url"
                                    type="url"
                                    value={newRepoUrl}
                                    onChange={(e) => setNewRepoUrl(e.target.value)}
                                    placeholder="https://github.com/owner/repo"
                                    disabled={adding}
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="repo-branch" className="text-sm font-medium">
                                    Branch
                                </Label>
                                <Input
                                    id="repo-branch"
                                    type="text"
                                    value={newBranch}
                                    onChange={(e) => setNewBranch(e.target.value)}
                                    placeholder="main"
                                    disabled={adding}
                                    className="h-9"
                                />
                            </div>
                            <div className="flex gap-2 justify-end">
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setShowAddForm(false)}
                                    disabled={adding}
                                >
                                    Cancel
                                </Button>
                                <Button
                                    size="sm"
                                    onClick={handleAddProject}
                                    disabled={adding || !newRepoUrl.trim()}
                                >
                                    {adding ? (
                                        <>
                                            <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                            Ingesting...
                                        </>
                                    ) : (
                                        "Add Project"
                                    )}
                                </Button>
                            </div>
                        </div>
                    ) : (
                        <Button
                            variant="outline"
                            className="w-full border-dashed"
                            onClick={() => setShowAddForm(true)}
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            Add Repository
                        </Button>
                    )}

                    {/* Projects List */}
                    {loading ? (
                        <div className="flex items-center justify-center py-8 text-muted-foreground">
                            <Loader2 className="h-5 w-5 animate-spin mr-2" />
                            Loading projects...
                        </div>
                    ) : projects.length === 0 ? (
                        <div className="text-center py-8 text-muted-foreground">
                            <GitBranch className="h-10 w-10 mx-auto mb-3 opacity-50" />
                            <p className="text-sm">No projects yet</p>
                            <p className="text-xs mt-1">
                                Add a GitHub repository to get started
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {projects.map((project) => (
                                <div
                                    key={project.id}
                                    className="border rounded-lg p-4 space-y-3 hover:border-primary/30 transition-colors"
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2">
                                                <h4 className="font-semibold text-sm truncate">
                                                    {project.repo_owner}/{project.repo_name}
                                                </h4>
                                                <a
                                                    href={project.repo_url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="text-muted-foreground hover:text-primary"
                                                >
                                                    <ExternalLink className="h-3.5 w-3.5" />
                                                </a>
                                            </div>
                                            <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                                                <span className="flex items-center gap-1">
                                                    <GitBranch className="h-3 w-3" />
                                                    {project.branch}
                                                </span>
                                                <span className="flex items-center gap-1">
                                                    <Clock className="h-3 w-3" />
                                                    Synced: {formatSyncTime(project.last_synced_at)}
                                                </span>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-1 ml-2">
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8"
                                                onClick={() => handleSync(project.id)}
                                                disabled={syncing === project.id}
                                                title="Sync now"
                                            >
                                                {syncing === project.id ? (
                                                    <Loader2 className="h-4 w-4 animate-spin" />
                                                ) : (
                                                    <RefreshCw className="h-4 w-4" />
                                                )}
                                            </Button>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8 text-destructive hover:text-destructive"
                                                onClick={() => handleDelete(project.id)}
                                                disabled={deleting === project.id}
                                                title="Delete project"
                                            >
                                                {deleting === project.id ? (
                                                    <Loader2 className="h-4 w-4 animate-spin" />
                                                ) : (
                                                    <Trash2 className="h-4 w-4" />
                                                )}
                                            </Button>
                                        </div>
                                    </div>

                                    {/* Report Generation */}
                                    <div className="flex items-center gap-2 pt-1 border-t border-border/50">
                                        <Select
                                            value={reportType}
                                            onValueChange={setReportType}
                                        >
                                            <SelectTrigger className="h-8 text-xs w-[140px]">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="full">Full Report</SelectItem>
                                                <SelectItem value="architecture">
                                                    Architecture
                                                </SelectItem>
                                                <SelectItem value="dependencies">
                                                    Dependencies
                                                </SelectItem>
                                                <SelectItem value="code_quality">
                                                    Code Quality
                                                </SelectItem>
                                            </SelectContent>
                                        </Select>
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            className="h-8 text-xs"
                                            onClick={() => handleGenerateReport(project.id)}
                                            disabled={reportLoading === project.id}
                                        >
                                            {reportLoading === project.id ? (
                                                <>
                                                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                                                    Generating...
                                                </>
                                            ) : (
                                                <>
                                                    <FileText className="h-3 w-3 mr-1" />
                                                    Generate Report
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Report Display */}
                    {reportContent && (
                        <div className="border rounded-lg p-4 space-y-2 bg-muted/20">
                            <div className="flex items-center justify-between">
                                <h4 className="font-semibold text-sm flex items-center gap-2">
                                    <FileText className="h-4 w-4 text-primary" />
                                    Generated Report
                                </h4>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-7 text-xs"
                                    onClick={() => setReportContent(null)}
                                >
                                    Close
                                </Button>
                            </div>
                            <div className="prose prose-sm dark:prose-invert max-w-none max-h-[400px] overflow-y-auto text-sm whitespace-pre-wrap">
                                {reportContent}
                            </div>
                        </div>
                    )}
                </div>

                <DialogFooter>
                    <Button variant="ghost" onClick={() => onOpenChange(false)}>
                        Close
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
