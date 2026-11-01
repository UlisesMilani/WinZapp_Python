# Arabic localization

Arabic (`ar`, displayed as `العربية`) is an initial community translation of
the interface and the Windows installer/uninstaller. Keep the product name
**WinZapp** in English in Arabic text, including inflected sentences. The PO
at `translations/ar/LC_MESSAGES/winzapp.po` is authoritative.

## Translation and review status

The initial translation was produced with AI assistance through OpenCode.
Additional source updates and obvious mixed-language wording were reviewed
while preparing the contribution. The contributor tried the portable Arabic
preview and reported that the translation looked good and was provisionally
acceptable. This is initial user acceptance, not a comprehensive linguistic,
right-to-left layout, or screen-reader certification. Further Arabic-speaking
review and accessibility feedback are welcome.

The complete usage guide has not been translated. Help opens an Arabic
introduction that explicitly links to the complete English guide. Its source
is `docs/user-guide/arabic-introduction.html`; the normal guide generator copies
it to `client/data/help_guide/ar.html`, and the guide freshness check covers it.
The other seven complete guides retain their existing structure.

## Updating the translation

Run `translations-update` after changes to the English catalog, review new and
fuzzy Arabic entries, then run `translations-compile` and `translations-check`.
Preserve format fields, markup, keyboard mnemonics and existing terminology.
Do not mark changed wording reviewed solely because an automated check passes.
No automatic AI translation service, credentials, or scheduled workflow is
introduced by this contribution.

The existing singular-key translation API does not support gettext plural
entries. Adding Arabic plural handling or globally mirroring native wx controls
needs a separate tested change; this translation does not claim either feature.
