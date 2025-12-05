---
title: "Zed Extension for Camouflage"
tags: ["zed", "rust"]
ShowToc: false
---

I am using [Zed editor](https://zed.dev/) especially the [Vim mode](https://zed.dev/docs/vim) for quite some time. VSCode editor was my previous editor. The experience on VSCode was laggy and slow using [VSCodeVim](https://github.com/VSCodeVim/Vim). I'll blame my machine for that, rather than my not-so-great Vim skills.

I quite like the [Camouflage](https://marketplace.visualstudio.com/items?itemName=zeybek.camouflage) VSCode extension by [zeybek](https://marketplace.visualstudio.com/publishers/zeybek). This extension hides the sensitive environment variables in the `.env` file by hiding their values. No more "oops" moment.

This seems like a perfect opportunity to learn Rust - Create a similar extension for Zed editor.

---

> [!INFO]
> Project code: https://github.com/dudeperf3ct/zed-camouflage

After looking around, I found out that [slash command](https://zed.dev/docs/extensions/slash-commands) approach should be used to implement this. The idea being adding `/mask` slash command, the contents of `.env` file would be masked.

> [!NOTE]
> The assistant panel in Zed is not intuitive. I had to click aimlessly to find "Assistant panel" where slash commands can be inserted.

> [!TIP]
> The assistant panel is located at the bottom right (or shortcut `Ctrl + ?`). In there click on the `+` followed by "New Text Thread" (or shortcut `ctrl + alt + n`).

**Attempt 1**: Slash command `run_slash_command`

The steps would be

```rust
impl zed::Extension for CamouflageExtension {
    fn run_slash_command(...) {
        // read file using worktree
        // masked = mask_env_contents(&contents)
        // return SlashCommandOutput{text: masked, sections: vec![]}
    }
}
```

1. Read the contents of `.env` file
2. Parse and mask the contents of the file
3. Write the output to the file `.env`

The problem with above is 

1. `Worktree` do not contain the files of current working directory.
2. `SlashCommandOutput` returns output as a text field in the panel itself.

**Attempt 2**: Abandon `Worktree` and pass the contents of `.env` as arguments to slash command and then perform masking

```rust
impl zed::Extension for CamouflageExtension {

    fn run_slash_command(...) -> ... {
        match command.name.as_str() {
            "mask" => self.handle_mask_command(args),
            _ => Err(format!("Unknown command: {}", command.name)),
        }
    }
}

impl CamouflageExtension {
    fn handle_mask_command(&self, args: Vec<String>) -> ... {
        // Check if args empty
        // Join all args into single string
        // Mask the content
        // Return masked output
    } 
}
```

This worked but that's not what I wanted. You have to navigate to "Assistant panel" and type `/mask` command with contents of `.env`, lol.

```bash
/mask API_KEY=secret123
```

Output

```bash
🔒 Masked Environment Variables:

API_KEY=***MASKED***
```

Looks like there's a similar open issue asking for the same: [How to read file contents in a custom slash command? #17714](https://github.com/zed-industries/zed/discussions/17714).

> [!CAUTION]
> Extensions run inside of the Wasm sandbox and can currently only perform arbitrary file I/O within the extension working directory.
