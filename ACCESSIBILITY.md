# Accessibility

LangScope exists to make text on screen readable to people who do not know its language, so accessibility matters to it. This page says what works today, what does not, and how to report a barrier. LangScope has not been audited against an accessibility standard such as WCAG, and it does not claim conformance with one.

## Supported environments

- Windows 10 and 11, with Python 3.11 or newer.
- Windows display scaling, including different scaling per monitor: the windows, the region selector and the overlay follow it.
- Mouse and keyboard input. Touch and pen input work only as far as Windows maps them to the mouse.

## What works

- **Keyboard use of the control panel.** The panel is built from standard Qt widgets: the language drop-downs, the **Select Region** and **Start**/**Stop** buttons and the **Desktop Overlay** and **Debug Mode** checkboxes. They can be reached with `Tab` and operated with the keyboard.
- **Assistive technology.** Qt exposes those standard widgets to Windows UI Automation, so screen readers can name and operate them.
- **Contrast.** Translated text is drawn in light text on a near-black background, with a contrast ratio of about 17:1, both in the app and in the desktop overlay. Body text in the control panel is about 15:1 against its background.
- **Text size.** The translation is sized to the height of the original text and shrinks only as far as it must to fit the box.
- **Cancelling.** `Esc` or a right-click cancels region selection.

## Known limitations

- **Region selection needs a mouse.** The capture region is drawn by dragging; there is no keyboard way to set it yet.
- **Translations are not available to screen readers.** The translated text is painted into the live feed and the desktop overlay as pixels, so a screen reader cannot read it. This is the largest gap for blind and low-vision users.
- **Some secondary text has low contrast.** The status line (about 4.4:1), the badges (about 4.1:1) and the white labels on the blue buttons (about 3.4:1) are below the 4.5:1 that WCAG asks for small text.
- **The theme is fixed.** LangScope uses its own light theme and does not follow Windows high-contrast themes or dark mode.
- **Font sizes in the panel are fixed.** They follow Windows display scaling, but not the separate Windows text-size setting.
- **The overlay is click-through and cannot take focus**, so it cannot be reached with the keyboard or assistive technology. It can be turned off with the **Desktop Overlay** checkbox.
- **Updates are slow on modest hardware.** A pass of OCR can take a second or more (see the [Benchmark](README.md#benchmark)), so the translation of changing text arrives with a delay.

## Reporting a barrier

If something in LangScope keeps you from using it, please [open an issue](https://github.com/MohamedxChaouachi/LangScope/issues/new/choose) with the bug report template and start the title with "Accessibility:". Say what you were trying to do, what got in the way, and which assistive technology and Windows version you use, if any. Accessibility issues are treated as bugs.
