import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select"
import { useState } from "react"
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"

export function Settings() {
  const getInitial = () => {
    if (typeof window === "undefined") {
      return { apiKey: "", baseUrl: "", modelId: "", storage: "local" as const }
    }
    try {
      const sessionApiKey = sessionStorage.getItem("PICOLLM_API_KEY")
      const localApiKey = localStorage.getItem("PICOLLM_API_KEY")
      const sessionBaseUrl = sessionStorage.getItem("PICOLLM_BASE_URL")
      const localBaseUrl = localStorage.getItem("PICOLLM_BASE_URL")
      const sessionModelId = sessionStorage.getItem("PICOLLM_MODEL")
      const localModelId = localStorage.getItem("PICOLLM_MODEL")
      if (sessionApiKey || sessionBaseUrl || sessionModelId) {
        return {
          apiKey: sessionApiKey || "",
          baseUrl: sessionBaseUrl || "",
          modelId: sessionModelId || "",
          storage: "session" as const,
        }
      }
      if (localApiKey || localBaseUrl || localModelId) {
        return {
          apiKey: localApiKey || "",
          baseUrl: localBaseUrl || "",
          modelId: localModelId || "",
          storage: "local" as const,
        }
      }
    } catch {
      // ignore
    }
    return { apiKey: "", baseUrl: "", modelId: "", storage: "local" as const }
  }

  const initial = getInitial()
  const [apiKey, setApiKey] = useState<string>(initial.apiKey)
  const [baseUrl, setBaseUrl] = useState<string>(initial.baseUrl)
  const [modelId, setModelId] = useState<string>(initial.modelId)
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle")
  const [storage, setStorage] = useState<"local" | "session">(initial.storage)

  async function saveSettings() {
    setStatus("saving")
    try {
      try {
        if (storage === "local") {
          localStorage.setItem("PICOLLM_API_KEY", apiKey)
          localStorage.setItem("PICOLLM_BASE_URL", baseUrl)
          localStorage.setItem("PICOLLM_MODEL", modelId)
          sessionStorage.removeItem("PICOLLM_API_KEY")
          sessionStorage.removeItem("PICOLLM_BASE_URL")
          sessionStorage.removeItem("PICOLLM_MODEL")
        } else {
          sessionStorage.setItem("PICOLLM_API_KEY", apiKey)
          sessionStorage.setItem("PICOLLM_BASE_URL", baseUrl)
          sessionStorage.setItem("PICOLLM_MODEL", modelId)
          localStorage.removeItem("PICOLLM_API_KEY")
          localStorage.removeItem("PICOLLM_BASE_URL")
          localStorage.removeItem("PICOLLM_MODEL")
        }
      } catch {
        // ignore storage errors
      }
      setStatus("saved")
    } catch (err) {
      console.error(err)
      setStatus("error")
    }
    setTimeout(() => setStatus("idle"), 2000)
  }

  function clearSettings() {
    try {
      localStorage.removeItem("PICOLLM_API_KEY")
      localStorage.removeItem("PICOLLM_BASE_URL")
      localStorage.removeItem("PICOLLM_MODEL")
      sessionStorage.removeItem("PICOLLM_API_KEY")
      sessionStorage.removeItem("PICOLLM_BASE_URL")
      sessionStorage.removeItem("PICOLLM_MODEL")
      setApiKey("")
      setBaseUrl("")
      setModelId("")
      setStorage("local")
      setStatus("idle")
    } catch {}
  }
  
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline">Settings</Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Model settings</SheetTitle>
          <SheetDescription>
            Point the app at a running picoLLM OpenAI-compatible endpoint.
          </SheetDescription>
        </SheetHeader>
        <div className="grid flex-1 auto-rows-min gap-6 px-4">
          <div className="grid gap-3">
            <Label htmlFor="base-url">PICOLLM_BASE_URL</Label>
            <Input
              id="base-url"
              value={baseUrl}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBaseUrl(e.target.value)}
              placeholder="http://127.0.0.1:8008/v1"
            />
          </div>
          <div className="grid gap-3">
            <Label htmlFor="model-id">PICOLLM_MODEL</Label>
            <Input
              id="model-id"
              value={modelId}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setModelId(e.target.value)}
              placeholder="picollm-chat"
            />
          </div>
          <div className="grid gap-3">
            <Label htmlFor="api-key">PICOLLM_API_KEY</Label>
            <Input
              type="password"
              id="api-key"
              value={apiKey}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setApiKey(e.target.value)}
              placeholder="Optional for local demos"
            />
          </div>
        </div>
        <SheetFooter>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <label className="text-sm">Store in:</label>
              <Select value={storage} onValueChange={(v) => setStorage(v as "local" | "session") }>
                <SelectTrigger className="w-56">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="local">Local Storage (persist)</SelectItem>
                  <SelectItem value="session">Session Storage (current tab)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button type="button" onClick={saveSettings} disabled={status === "saving"}>
              {status === "saving" ? "Saving..." : status === "saved" ? "Saved" : "Save changes"}
            </Button>

            <Button type="button" variant="outline" onClick={clearSettings}>
              Clear
            </Button>

            <SheetClose asChild>
              <Button variant="outline">Close</Button>
            </SheetClose>
          </div>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
