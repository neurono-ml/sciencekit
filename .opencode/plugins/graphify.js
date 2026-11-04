// graphify OpenCode plugin (V2 plugin API)
// Injects a knowledge graph reminder before shell tool calls when the graph exists.
import { Plugin } from "@opencode/plugin";
import { existsSync } from "node:fs";
import { join } from "node:path";

const REMINDER =
  '[graphify] knowledge graph at graphify-out/. For focused questions, run `graphify query "<question>"` ' +
  "(scoped subgraph, usually much smaller than GRAPH_REPORT.md) instead of grepping raw files. " +
  "Read GRAPH_REPORT.md only for broad architecture context.";

export default Plugin.define({
  id: "graphify",
  async setup(ctx) {
    let reminded = false;

    await ctx.tool.hook("execute.before", (event) => {
      if (reminded) return;
      if (event.tool !== "shell" && event.tool !== "bash") return;

      const input = event.input;
      if (!input || typeof input.command !== "string") return;
      if (!existsSync(join(ctx.location.directory, "graphify-out", "graph.json"))) return;

      // Wrap the reminder in single quotes so the backticks it contains are
      // literal text and never trigger shell command substitution.
      const quoted = `'${REMINDER.replaceAll("'", "'\\''")}'`;
      input.command = `echo ${quoted} && ${input.command}`;
      reminded = true;
    });
  },
});
