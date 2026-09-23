# Pinned Git LFS hydration

`tools/hydrate_pinned_lfs.py` packages the Git-LFS behavior learned while
qualifying Coconut's historical GSM data source into a runner-neutral command.

It exists because an exact historical revision can contain an LFS pointer while
predating the repository's later `.gitattributes` tracking rule. A naive exact
checkout therefore leaves the pointer unhydrated.

## Method

The tool accepts exactly one LFS object and optional ordinary companion files.

It:

1. requires an exact 40-character Git revision;
2. restricts the source to a GitHub `owner/repo` HTTPS clone;
3. clones without checkout and with automatic smudging disabled;
4. installs Git LFS locally in that disposable clone;
5. writes only the requested LFS path into local
   `.git/info/attributes`;
6. sparse-checks out only the requested paths at the exact revision;
7. reads the committed pointer directly with `git show`;
8. verifies its expected SHA-256 OID and byte size;
9. verifies the hydrated working-tree object's SHA-256 and byte size;
10. hashes optional companion files and emits a JSON receipt.

It never invokes a shell. The local attribute changes no repository content.

## Coconut GSM example

The qualified object is:

- repository: `da03/Internalize_CoT_Step_by_Step`;
- revision: `e06a32ee5e4cd117171daeb4755d2a97ece62761`;
- path: `data/gsm8k/train.txt`;
- OID: `0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c`;
- size: 87,805,358 bytes.

The same command can run on GHA or a sovereign/local host. Git LFS transport is
therefore no longer encoded only in Actions YAML.

## Boundary

This is a source-hydration primitive, not a cache, artifact registry, dataset
manager, or scheduler. Those responsibilities stay outside this tool.
