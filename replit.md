# PDF Question Extractor

## Overview

This is a full-stack web application that extracts questions from PDF documents, particularly textbooks, and converts them to LaTeX format. The application features a React frontend with shadcn/ui components and an Express.js backend with PostgreSQL database integration. The system is designed to process PDF files, analyze their content using AI/ML techniques, and generate formatted question sets for educational purposes.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **UI Library**: shadcn/ui components built on Radix UI primitives
- **Styling**: Tailwind CSS with CSS variables for theming
- **State Management**: TanStack Query for server state management
- **Routing**: React Router DOM for client-side routing
- **Build Tool**: Vite with hot module replacement

### Backend Architecture
- **Runtime**: Node.js with TypeScript
- **Framework**: Express.js with middleware for JSON parsing and request logging
- **Database ORM**: Drizzle ORM for type-safe database operations
- **Session Management**: connect-pg-simple for PostgreSQL session storage
- **Development**: tsx for TypeScript execution and hot reloading

### Data Storage
- **Database**: PostgreSQL with Neon serverless integration
- **Schema Management**: Drizzle Kit for migrations and schema definition
- **Storage Interface**: Abstracted storage layer with both memory and database implementations
- **Schema Design**: Users table with username/password authentication

### Authentication & Authorization
- **User Management**: Basic user registration and authentication system
- **Session Storage**: PostgreSQL-backed sessions via connect-pg-simple
- **Data Validation**: Zod schemas for request/response validation

### Development & Deployment
- **Development Server**: Vite dev server with proxy configuration
- **Production Build**: Static asset generation with Express.js API serving
- **Environment**: Environment variable configuration for database connections
- **Code Quality**: TypeScript strict mode with path aliases for clean imports

### API Structure
- RESTful API design with `/api` prefix for all backend routes
- Centralized error handling middleware
- Request/response logging with performance metrics
- Type-safe route handlers with shared schema definitions

## External Dependencies

### Core Dependencies
- **@neondatabase/serverless**: Neon PostgreSQL serverless driver for database connectivity
- **drizzle-orm & drizzle-kit**: Type-safe ORM and migration toolkit
- **@tanstack/react-query**: Server state management and caching
- **react-router-dom**: Client-side routing

### UI Components
- **@radix-ui/***: Comprehensive set of headless UI primitives (accordion, dialog, dropdown, etc.)
- **lucide-react**: Icon library for consistent iconography
- **class-variance-authority**: Utility for managing component variants
- **tailwindcss**: Utility-first CSS framework

### Development Tools
- **vite**: Fast build tool with HMR
- **tsx**: TypeScript execution engine
- **@replit/vite-plugin-runtime-error-modal**: Development error overlay
- **@replit/vite-plugin-cartographer**: Replit-specific development enhancements

### Form & Validation
- **react-hook-form**: Performant form library
- **@hookform/resolvers**: Form validation resolvers
- **zod**: Schema validation library
- **drizzle-zod**: Integration between Drizzle and Zod

### Utility Libraries
- **date-fns**: Date manipulation utilities
- **clsx & tailwind-merge**: Utility for conditional CSS classes
- **cmdk**: Command palette component