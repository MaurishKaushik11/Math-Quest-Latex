import { Moon, Sun } from "lucide-react"
import { Button } from "./button"
import { useTheme } from "../theme-provider"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./dropdown-menu"

export function ThemeToggle() {
  return (
    <div className="flex items-center gap-2" data-testid="theme-toggle">
      <div className="w-8 h-8 rounded-full bg-gradient-to-r from-green-400 to-orange-400 flex items-center justify-center shadow-md">
        <Sun className="h-4 w-4 text-white" />
      </div>
      <span className="text-sm font-medium" style={{color: 'hsl(0, 0%, 5%)'}}>Pastel Theme</span>
    </div>
  )
}