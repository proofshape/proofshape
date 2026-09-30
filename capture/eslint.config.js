// @ts-check
import { defineConfig } from "eslint/config";
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

// Type-aware rules (strictTypeChecked) are scoped to **/*.ts only, so this file itself — plain
// JS tooling config, not part of the capture/ tsconfig project — never needs type information.
export default defineConfig([
  { ignores: ["dist/**", "node_modules/**"] },
  eslint.configs.recommended,
  {
    files: ["**/*.ts"],
    extends: [tseslint.configs.strictTypeChecked],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
]);
