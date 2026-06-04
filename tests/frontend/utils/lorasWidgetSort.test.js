import { beforeEach, describe, expect, it, vi } from "vitest";

const { APP_MODULE, UTILS_MODULE } = vi.hoisted(() => ({
  APP_MODULE: new URL("../../../scripts/app.js", import.meta.url).pathname,
  UTILS_MODULE: new URL("../../../web/comfyui/loras_widget_utils.js", import.meta.url).pathname,
}));

vi.mock(APP_MODULE, () => ({
  app: {
    extensionManager: { toast: { add: vi.fn() } },
  },
}));

describe("loras widget folder helpers", () => {
  let getLoraFolder;
  let getLoraBasename;
  let sortLorasByFolder;

  beforeEach(async () => {
    vi.resetModules();
    ({ getLoraFolder, getLoraBasename, sortLorasByFolder } = await import(UTILS_MODULE));
  });

  describe("getLoraBasename", () => {
    it("returns the file name portion of a path", () => {
      expect(getLoraBasename("characters/anime/myLora")).toBe("myLora");
    });

    it("returns the name unchanged when there is no folder", () => {
      expect(getLoraBasename("myLora")).toBe("myLora");
    });

    it("normalizes Windows-style backslashes", () => {
      expect(getLoraBasename("characters\\anime\\myLora")).toBe("myLora");
    });
  });

  describe("getLoraFolder", () => {
    it("returns the directory portion of a path name", () => {
      expect(getLoraFolder("characters/anime/myLora")).toBe("characters/anime");
    });

    it("returns an empty string when there is no folder", () => {
      expect(getLoraFolder("myLora")).toBe("");
    });

    it("normalizes Windows-style backslashes", () => {
      expect(getLoraFolder("characters\\anime\\myLora")).toBe("characters/anime");
    });

    it("handles empty/invalid input", () => {
      expect(getLoraFolder("")).toBe("");
      expect(getLoraFolder(undefined)).toBe("");
      expect(getLoraFolder(null)).toBe("");
    });
  });

  describe("sortLorasByFolder", () => {
    it("groups by folder, then orders by file name, root first", () => {
      const input = [
        { name: "characters/b" },
        { name: "rootLora" },
        { name: "characters/a" },
        { name: "styles/x" },
      ];
      const sorted = sortLorasByFolder(input);
      expect(sorted.map((l) => l.name)).toEqual([
        "rootLora",
        "characters/a",
        "characters/b",
        "styles/x",
      ]);
    });

    it("sorts case-insensitively and with natural number ordering", () => {
      const input = [
        { name: "f/Lora10" },
        { name: "f/lora2" },
        { name: "f/Lora1" },
      ];
      const sorted = sortLorasByFolder(input);
      expect(sorted.map((l) => l.name)).toEqual(["f/Lora1", "f/lora2", "f/Lora10"]);
    });

    it("preserves the lora object fields and does not mutate the input", () => {
      const input = [
        { name: "b/two", strength: 0.5, active: true },
        { name: "a/one", strength: 0.8, active: false },
      ];
      const sorted = sortLorasByFolder(input);
      expect(sorted[0]).toEqual({ name: "a/one", strength: 0.8, active: false });
      expect(input.map((l) => l.name)).toEqual(["b/two", "a/one"]);
    });
  });
});
