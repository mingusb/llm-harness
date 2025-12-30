---
title: Installation
taxonomy:
    category: docs
---

In order to use Select2, you must include the compiled JavaScript and CSS files on your website. There are multiple options for including these pre-compiled files, also known as a **distribution**, in your website or application.

## Using Select2 from a CDN

A CDN (content delivery network) is the fastest way to get up and running with Select2!

Select2 is hosted on both the [jsDelivr](https://www.jsdelivr.com/package/npm/select2) and [cdnjs](https://cdnjs.com/libraries/select2) CDNs. Simply include the following lines of code in the `<head>` section of your page:

```
<link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>
```

>>> <i class="fa fa-info-circle"></i> Immediately following a new release, it takes some time for CDNs to catch up and get the new versions live on the CDN.

## Installing with Bower

Select2 is available on Bower.  Add the following to your `bower.json` file and then run `bower install`:

```
"dependencies": {
    "select2": "~4.0"
}
```

Or, run `bower install select2` from your project directory.

The precompiled distribution files will be available in `vendor/select2/dist/css/` and `vendor/select2/dist/js/`, relative to your project directory. Include them in your page:

```
<link href="vendor/select2/dist/css/select2.min.css" rel="stylesheet" />
<script src="vendor/select2/dist/js/select2.min.js"></script>
```

## Manual installation

We strongly recommend that you use either a CDN or a package manager like Bower or npm. This will make it easier for you to deploy your project in different environments, and easily update Select2 when new versions are released. Nonetheless if you prefer to integrate Select2 into your project manually, you can [download the release of your choice](https://github.com/select2/select2/tags) from GitHub and copy the files from the `dist` directory into your project.

Include the compiled files in your page:

```
<link href="path/to/select2.min.css" rel="stylesheet" />
<script src="path/to/select2.min.js"></script>
```
---
title: Basic usage
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

## Single select boxes

Select2 was designed to be a replacement for the standard `<select>` box that is displayed by the browser.  By default it supports all options and operations that are available in a standard select box, but with added flexibility.

Select2 can take a regular select box like this...

<select class="js-states form-control"></select>

and turn it into this...

<div class="s2-example">
    <select class="js-example-basic-single js-states form-control"></select>
</div>

```
<select class="js-example-basic-single" name="state">
  <option value="AL">Alabama</option>
    ...
  <option value="WY">Wyoming</option>
</select>
```

<script type="text/javascript" class="js-code-example-basic-single">
$(document).ready(function() {
    $('.js-example-basic-single').select2();
});
</script>

Select2 will register itself as a jQuery function if you use any of the distribution builds, so you can call `.select2()` on any jQuery selector where you would like to initialize Select2.

```
// In your Javascript (external .js resource or <script> tag)
$(document).ready(function() {
    $('.js-example-basic-single').select2();
});
```

>>>>>> The DOM cannot be safely manipulated until it is "ready".  To make sure that your DOM is ready before the browser initializes the Select2 control, wrap your code in a [`$(document).ready()`](https://learn.jquery.com/using-jquery-core/document-ready/) block.  Only one `$(document).ready()` block is needed per page.

## Multi-select boxes (pillbox)

Select2 also supports multi-value select boxes. The select below is declared with the `multiple` attribute.

<div class="s2-example">
  <p>
    <select class="js-example-basic-multiple js-states form-control" multiple="multiple"></select>
  </p>
</div>

**In your HTML:**

```
<select class="js-example-basic-multiple" name="states[]" multiple="multiple">
  <option value="AL">Alabama</option>
    ...
  <option value="WY">Wyoming</option>
</select>
```

**In your Javascript (external `.js` resource or `<script>` tag):**

```
$(document).ready(function() {
    $('.js-example-basic-multiple').select2();
});
```

<script type="text/javascript">
  $.fn.select2.amd.require([
    "select2/core",
    "select2/utils"
  ], function (Select2, Utils, oldMatcher) {
    var $basicSingle = $(".js-example-basic-single");
    var $basicMultiple = $(".js-example-basic-multiple");

    $.fn.select2.defaults.set("width", "100%");

    $basicSingle.select2();
    $basicMultiple.select2();

    function formatState (state) {
      if (!state.id) {
        return state.text;
      }
      var $state = $(
        '<span>' +
          '<img src="vendor/images/flags/' +
            state.element.value.toLowerCase() +
          '.png" class="img-flag" /> ' +
          state.text +
        '</span>'
      );
      return $state;
    };
  });

</script>
---
title: Builds and modules
taxonomy:
    category: docs
process:
    twig: true
---

## The different Select2 builds

Select2 provides multiple builds that are tailored to different
environments where it is going to be used. If you think you need to use
Select2 in a nonstandard environment, like when you are using AMD, you
should read over the list below.

<table class="table table-bordered table-striped">
  <thead>
    <tr>
      <th>Build name</th>
      <th>When you should use it</th>
    </tr>
  </thead>
  <tbody>
    <tr id="builds-standard">
      <td>
        Standard (<code>select2.js</code> / <code>select2.min.js</code>)
      </td>
      <td>
        This is the build that most people should be using for Select2. It
        includes the most commonly used features.
      </td>
    </tr>
    <tr id="builds-full">
      <td>
        Full (<code>select2.full.js</code> / <code>select2.full.min.js</code>)
      </td>
      <td>
        You should only use this build if you need the recommended includes like <a href="https://github.com/jquery/jquery-mousewheel">jquery.mousewheel</a>
      </td>
    </tr>
  </tbody>
</table>

## Using Select2 with AMD or CommonJS loaders

Select2 should work with most AMD- or CommonJS-compliant module loaders, including [RequireJS](http://requirejs.org/) and [almond](https://github.com/jrburke/almond). Select2 ships with a modified version of the [UMD jQuery template](https://github.com/umdjs/umd/blob/f208d385768ed30cd0025d5415997075345cd1c0/templates/jqueryPlugin.js) that supports both CommonJS and AMD environments.

### Configuration

For most AMD and CommonJS setups, the location of the data files used by Select2 will be automatically determined and handled without you needing to do anything.

Select2 internally uses AMD and the r.js build tool to build the files located in the `dist` folder. These are built using the files in the `src` folder, so _you can_ just point your modules to the Select2 source and load in `jquery.select2` or `select2/core` when you want to use Select2. The files located in the `dist` folder are also AMD-compatible, so you can point to that file if you want to load in all of the default Select2 modules.

If you are using Select2 in a build environment where preexisting module names are changed during a build step, Select2 may not be able to find optional language files. You can manually set the prefixes to use for these files using the `amdLanguageBase` options.

```
$.fn.select2.defaults.set('amdLanguageBase', 'select2/i18n/');
```

#### `amdLanguageBase`

Specifies the base AMD loader language path to be used for select2 language file resolution. This option typically doesn't need to be changed, but is available for situations where module names may change as a result of certain build environments.

>>> Due to [a bug in older versions](https://github.com/jrburke/requirejs/issues/1342) of the r.js build tool, Select2 was sometimes placed before jQuery in the compiled build file. Because of this, Select2 will trigger an error because it won't be able to find or load jQuery.  By upgrading to version 2.1.18 or higher of the r.js build tool, you will be able to fix the issue.
---
title: Getting Started
taxonomy:
    category: docs
process:
    twig: true
twig_first: true
---

![Select2 logo](/images/logo.png)

# Select2

The jQuery replacement for select boxes

<div class="s2-docs-featurette">
    <a class="button" href="https://github.com/select2/select2/releases">Releases</a>
    <a class="button" href="https://forums.select2.org">Forums</a>
    <a class="button" href="https://github.com/select2/select2">GitHub</a>
</div>

Select2 gives you a customizable select box with support for searching, tagging, remote data sets, infinite scrolling, and many other highly used options.

<div class="s2-docs-featurette">
    <div class="grid">
      <div class="size-1-3 size-tablet-1-2">
          <i class="fa fa-language fa-4x"></i>
          <h4>In your language</h4>
          <p>
            Select2 comes with support for
            <a href="{{base_url_absolute}}/i18n#rtl-support">RTL environments</a>,
            <a href="{{base_url_absolute}}/i18n#diacritics">searching with diacritics</a> and
            <a href="{{base_url_absolute}}/i18n#">over 40 languages</a> built-in.
          </p>
      </div>

      <div class="size-1-3 size-tablet-1-2">
          <i class="fa fa-database fa-4x"></i>
          <h4>Remote data support</h4>
          <p>
            <a href="{{base_url_absolute}}/data-sources/ajax">Using AJAX</a> you can efficiently
            search large lists of items.
          </p>
      </div>

      <div class="size-1-3 size-tablet-1-2">
          <i class="fa fa-paint-brush fa-4x"></i>
          <h4>Theming</h4>
          <p>
            Fully skinnable, CSS built with Sass and an
            <a href="https://github.com/ttskch/select2-bootstrap4-theme">optional theme for Bootstrap 4</a>.
          </p>
      </div>
    </div>

    <div class="grid">
      <div class="size-1-3 size-tablet-1-2">
        <i class="fa fa-plug fa-4x"></i>
        <h4>Fully extensible</h4>
        <p>
          The <a href="{{base_url_absolute}}/advanced">plugin system</a>
          allows you to easily customize Select2 to work exactly how you want it
          to.
        </p>
      </div>

      <div class="size-1-3 size-tablet-1-2">
        <i class="fa fa-tag fa-4x"></i>
        <h4>Dynamic item creation</h4>
        <p>
          Allow users to type in a new option and
          <a href="{{base_url_absolute}}/tagging">add it on the fly</a>.
        </p>
      </div>

      <div class="size-1-3 size-tablet-1-2">
        <i class="fa fa-html5 fa-4x"></i>
        <h4>Full browser support</h4>
        <p>Support for both modern and legacy browsers is built-in, even including Internet Explorer 11.</p>
      </div>
    </div>
</div>---
title: Getting Help
metadata:
    description: How to get support, report a bug, or suggest a feature for Select2.
taxonomy:
    category: docs
---

## General support

Having trouble getting Select2 working on your website? Is it not working together with another plugin, even though you think it should? Select2 has a few communities that you can go to for help getting it all working together.

1. Join our [forums](https://forums.select2.org), graciously hosted by [NextGI](https://nextgi.com) and start a new topic.
2. Search [Stack Overflow](http://stackoverflow.com/questions/tagged/jquery-select2?sort=votes)  **carefully** for existing questions that might address your issue. If you need to open a new question, make sure to include the `jquery-select2` tag.
3. Ask in the `#select2` channel on `chat.freenode.net` or use the [web irc client.](https://webchat.freenode.net/?channels=select2) 

>>>> Do **NOT** use the GitHub issue tracker for general support and troubleshooting questions.  The issue tracker is **only** for bug reports with a [minimal, complete, and verifiable example](https://stackoverflow.com/help/mcve) and feature requests.  Use the forums instead.

## Reporting bugs

Found a problem with Select2? Feel free to open a ticket on the Select2 repository on GitHub, but you should keep a few things in mind:

1. Use the [GitHub issue search](https://github.com/select2/select2/search?q=&type=Issues) to check if your issue has already been reported.
2. Try to isolate your problem as much as possible.  Use [JS Bin](http://jsbin.com/goqagokoye/edit?html,js,output) to create a [minimal, verifiable, and complete](https://stackoverflow.com/help/mcve) example of the problem.
3. Once you are sure the issue is with Select2, and not a third party library, [open an issue](https://github.com/select2/select2/issues/new) with a description of the bug, and link to your jsbin example.

You can find more information on reporting bugs in the [contributing guide,](https://github.com/select2/select2/blob/master/CONTRIBUTING.md#reporting-bugs-with-select2) including tips on what information to include.

>>>>> If you are not conversationally proficient in English, do **NOT** post a machine translation (e.g. Google Translate) to GitHub. Get help in crafting your question, either via the [forums](https://forums.select2.org) or in [chat](https://webchat.freenode.net/?channels=select2).  If all else fails, you may post your bug report or feature request in your native language and we will tag it with `translation-needed` so that it can be properly translated.

## Requesting new features

New feature requests are usually requested by the [Select2 community on GitHub,](https://github.com/select2/select2/issues) and are often fulfilled by [fellow contributors.](https://github.com/select2/select2/blob/master/CONTRIBUTING.md)

1.  Use the [GitHub issue search](https://github.com/select2/select2/search?q=&type=Issues) to check if your feature has already been requested.
2.  Check if it hasn't already been implemented as a [third party plugin.](https://github.com/search?q=topic%3Aselect2&type=Repositories)
3.  Please make sure you are only requesting a single feature, and not a collection of smaller features.

You can find more information on requesting new features in the [contributing guide.](https://github.com/select2/select2/blob/master/CONTRIBUTING.md#requesting-features-in-select2)
---
title: Common problems
metadata:
    description: Commonly encountered issues when using Select2.
taxonomy:
    category: docs
---

### Select2 does not function properly when I use it inside a Bootstrap modal.

This issue occurs because Bootstrap modals tend to steal focus from other elements outside of the modal.  Since by default, Select2 [attaches the dropdown menu to the `<body>` element](/dropdown#dropdown-placement), it is considered "outside of the modal".

To avoid this problem, you may attach the dropdown to the modal itself with the [dropdownParent](/dropdown#dropdown-placement) setting:

```
<div id="myModal" class="modal fade" tabindex="-1" role="dialog" aria-hidden="true">
    ...
    <select id="mySelect2">
        ...
    </select>
    ...
</div>

...

<script>
    $('#mySelect2').select2({
        dropdownParent: $('#myModal')
    });
</script>
```

This will cause the dropdown to be attached to the modal, rather than the `<body>` element.

**Alternatively**, you may simply globally override Bootstrap's behavior:

```
// Do this before you initialize any of your modals
$.fn.modal.Constructor.prototype.enforceFocus = function() {};
```

See [this answer](https://stackoverflow.com/questions/18487056/select2-doesnt-work-when-embedded-in-a-bootstrap-modal/19574076#19574076) for more information.

### The dropdown becomes misaligned/displaced when using pinch-zoom.

See [#5048](https://github.com/select2/select2/issues/5048).  The problem is that some browsers' implementations of pinch-zoom affect the `body` element, which [Select2 attaches to by default](https://select2.org/dropdown#dropdown-placement), causing it to render incorrectly.

The solution is to use `dropdownParent` to attach the dropdown to a more specific element.
---
title: Troubleshooting
metadata:
    description: The chapter covers some common issues you may encounter with Select2, as well as where you can go to get help.
taxonomy:
    category: docs
---

# Troubleshooting

The chapter covers some common issues you may encounter with Select2, as well as where you can go to get help.---
title: Options
taxonomy:
    category: docs
---

This is a list of all the Select2 configuration options.

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `ajax` | object | `null` | Provides support for [ajax data sources](/data-sources/ajax). |
| `allowClear` | boolean | `false` | Provides support for [clearable selections](/selections#clearable-selections). |
| `amdLanguageBase` | string | `./i18n/` | See [Using Select2 with AMD or CommonJS loaders](/builds-and-modules#using-select2-with-amd-or-commonjs-loaders). |
| `closeOnSelect` | boolean | `true` | Controls whether the dropdown is [closed after a selection is made](/dropdown#forcing-the-dropdown-to-remain-open-after-selection). |
| `data` | array of objects | `null` | Allows rendering dropdown options from an [array](/data-sources/arrays). |
| `dataAdapter` | | `SelectAdapter` | Used to override the built-in [DataAdapter](/advanced/default-adapters/data). |
| `debug` | boolean | `false` | Enable debugging messages in the browser console. |
| `dir` | string | `ltr` | Sets the [`dir` attribute](https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/dir) on the selection and dropdown containers to indicate the direction of the text. |
| `disabled` | boolean | `false` | When set to `true`, the select control will be disabled. |
| `dropdownAdapter` | | `DropdownAdapter` | Used to override the built-in [DropdownAdapter](/advanced/default-adapters/dropdown) |
| `dropdownAutoWidth` | boolean | `false` | |
| `dropdownCssClass` | string | `''` | Adds additional CSS classes to the dropdown container. The helper `:all:` can be used to add all CSS classes present on the original `<select>` element. |
| `dropdownParent` | jQuery selector or DOM node | `$(document.body)` | Allows you to [customize placement](/dropdown#dropdown-placement) of the dropdown. |
| `escapeMarkup` | callback | `Utils.escapeMarkup` | Handles [automatic escaping of content rendered by custom templates](/dropdown#built-in-escaping). |
| `language` | string or object | `EnglishTranslation` | Specify the [language used for Select2 messages](/i18n#message-translations). |
| `matcher` | A callback taking search `params` and the `data` object. | | Handles custom [search matching](/searching#customizing-how-results-are-matched). |
| `maximumInputLength` | integer | `0` | [Maximum number of characters](/searching#maximum-search-term-length) that may be provided for a search term. |
| `maximumSelectionLength` | integer | `0` | The maximum number of items that may be selected in a multi-select control. If the value of this option is less than 1, the number of selected items will not be limited.
| `minimumInputLength` | integer | `0` | [Minimum number of characters required to start a search.](/searching#minimum-search-term-length) |
| `minimumResultsForSearch` | integer | `0` | The minimum number of results required to [display the search box](/searching#limiting-display-of-the-search-box-to-large-result-sets). |
| `multiple` | boolean | `false` | This option enables multi-select (pillbox) mode. Select2 will automatically map the value of the `multiple` HTML attribute to this option during initialization. |
| `placeholder` | string or object | `null` | Specifies the [placeholder](/placeholders) for the control. |
| `resultsAdapter` | | `ResultsAdapter` | Used to override the built-in [ResultsAdapter](/advanced/default-adapters/results). |
| `selectionAdapter` | | `SingleSelection` or `MultipleSelection`, depending on the value of `multiple`. | Used to override the built-in [SelectionAdapter](/advanced/default-adapters/selection). |
| `selectionCssClass` | string | `''` | Adds additional CSS classes to the selection container. The helper `:all:` can be used to add all CSS classes present on the original `<select>` element. |
| `selectOnClose` | boolean | `false` | Implements [automatic selection](/dropdown#automatic-selection) when the dropdown is closed. |
| `sorter` | callback | | |
| `tags` | boolean / array of objects | `false` | Used to enable [free text responses](/tagging). |
| `templateResult` | callback | | Customizes the way that [search results are rendered](/dropdown#templating). |
| `templateSelection` | callback | | Customizes the way that [selections are rendered](/selections#templating). |
| `theme` | string | `default` | Allows you to set the [theme](/appearance#themes). |
| `tokenizer` | callback | | A callback that handles [automatic tokenization of free-text entry](/tagging#automatic-tokenization-into-tags). |
| `tokenSeparators` | array | `null` | The list of characters that should be used as token separators. |
| `width` | string | `resolve` | Supports [customization of the container width](/appearance#container-width). |
| `scrollAfterSelect` | boolean | `false` | If `true`, resolves issue for multiselects using `closeOnSelect: false` that caused the list of results to scroll to the first selection after each select/unselect (see https://github.com/select2/select2/pull/5150). This behaviour was intentional to deal with infinite scroll UI issues (if you need this behavior, set `false`) but it created an issue with multiselect dropdown boxes of fixed length. |
---
title: Global defaults
taxonomy:
    category: docs
---

In some cases, you need to set the default options for all instances of Select2 in your web application. This is especially useful when you are migrating from past versions of Select2, or you are using non-standard options like [custom AMD builds](/getting-started/builds-and-modules#using-select2-with-amd-or-commonjs-loaders). Select2 exposes the default options through `$.fn.select2.defaults`, which allows you to set them globally.

When setting options globally, any past defaults that have been set will be overridden. Default options are only used when an option is requested that has not been set during initialization.

You can set default options by calling `$.fn.select2.defaults.set("key", "value")`.  For example:

```
$.fn.select2.defaults.set("theme", "classic");
```

## Nested options

To set a default values for cache, use the same notation used for [HTML `data-*` attributes](/configuration/data-attributes).  Two dashes (`--`) will be replaced by a level of nesting, and a single dash (`-`) will convert the key to a camelCase string:

```
$.fn.select2.defaults.set("ajax--cache", false);
```

## Resetting default options

You can reset the default options to their initial values by calling

```
$.fn.select2.defaults.reset();
```
---
title: data-* attributes
taxonomy:
    category: docs
---

It is recommended that you declare your configuration options by [passing in an object](/configuration) when initializing Select2.  However, you may also define your configuration options by using the HTML5 `data-*` attributes, which will override any options set when initializing Select2 and any [defaults](/configuration/defaults).

```
<select data-placeholder="Select a state">
  <option value="AL">Alabama</option>
    ...
  <option value="WY">Wyoming</option>
</select>
```

>>> Some options are not supported as `data-*`, for example `disabled` as it's not a Javascript option, but it's an HTML [attribute](/configuration/options-api).

## Nested (subkey) options

Sometimes, you have options that are nested under a top-level option.  For example, the options under the `ajax` option:

```
$(".js-example-data-ajax").select2({
  ajax: {
    url: "http://example.org/api/test",
    cache: false
  }
});
```

To write these options as `data-*` attributes, each level of nesting should be separated by two dashes (`--`):

```
<select data-ajax--url="http://example.org/api/test" data-ajax--cache="true">
    ...
</select>
```

The value of the option is subject to jQuery's [parsing rules](https://api.jquery.com/data/#data-html5) for HTML5 data attributes.

>>> Due to [a jQuery bug](https://github.com/jquery/jquery/issues/2070), nested options using `data-*` attributes [do not work in jQuery 1.x](https://github.com/select2/select2/issues/2969).

## `camelCase` options

HTML data attributes are case-insensitive, so any options which contain capital letters will be parsed as if they were all lowercase. Because Select2 has many options which are camelCase, where words are separated by uppercase letters, you must write these options out with dashes instead. So an option that would normally be called `allowClear` should instead be defined as `allow-clear`.

This means that declaring your `<select>` tag as...

```
<select data-tags="true" data-placeholder="Select an option" data-allow-clear="true">
    ...
</select>
```

Will be interpreted the same as initializing Select2 as...

```
$("select").select2({
  tags: "true",
  placeholder: "Select an option",
  allowClear: true
});
```
---
title: Configuration
taxonomy:
    category: docs
---

To configure custom options when you initialize Select2, simply pass an object in your call to `.select2()`:

```
$('.js-example-basic-single').select2({
  placeholder: 'Select an option'
});
```
---
title: Appearance
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

The appearance of your Select2 controls can be customized via the standard HTML attributes for `<select>` elements, as well as various [configuration options](/configuration).

## Disabling a Select2 control

Select2 will respond to the <code>disabled</code> attribute on `<select>` elements. You can also initialize Select2 with `disabled: true` to get the same effect.

<div class="s2-example">
  <p>
    <select class="js-example-disabled js-states form-control" disabled="disabled"></select>
  </p>

  <p>
    <select class="js-example-disabled-multi js-states form-control" multiple="multiple" disabled="disabled"></select>
  </p>
  <div class="btn-group btn-group-sm" role="group" aria-label="Programmatic enabling and disabling">
    <button type="button" class="js-programmatic-enable btn btn-default">
      Enable
    </button>
    <button type="button" class="js-programmatic-disable btn btn-default">
      Disable
    </button>
  </div>
</div>

<pre data-fill-from=".js-code-disabled"></pre>

<script type="text/javascript" class="js-code-disabled">

$(".js-example-disabled").select2();
$(".js-example-disabled-multi").select2();
  
$(".js-programmatic-enable").on("click", function () {
  $(".js-example-disabled").prop("disabled", false);
  $(".js-example-disabled-multi").prop("disabled", false);
});

$(".js-programmatic-disable").on("click", function () {
  $(".js-example-disabled").prop("disabled", true);
  $(".js-example-disabled-multi").prop("disabled", true);
});

</script>

## Labels

You can, and should, use a `<label>` with Select2, just like any other `<select>` element.

<div class="s2-example">
  <p>
    <label for="id_label_single">
      Click this to focus the single select element
      <select class="js-example-basic-single js-states form-control" id="id_label_single"></select>
    </label>
  </p>
  <p>
    <label for="id_label_multiple">
      Click this to focus the multiple select element
      <select class="js-example-basic-multiple js-states form-control" id="id_label_multiple" multiple="multiple"></select>
    </label>
  </p>
</div>

```
<label for="id_label_single">
  Click this to highlight the single select element

  <select class="js-example-basic-single js-states form-control" id="id_label_single"></select>
</label>

<label for="id_label_multiple">
  Click this to highlight the multiple select element

  <select class="js-example-basic-multiple js-states form-control" id="id_label_multiple" multiple="multiple"></select>
</label>
```

<script type="text/javascript">
  $.fn.select2.amd.require([
    "select2/core",
    "select2/utils"
  ], function (Select2, Utils, oldMatcher) {
    var $basicSingle = $(".js-example-basic-single");
    var $basicMultiple = $(".js-example-basic-multiple");

    $.fn.select2.defaults.set("width", "100%");

    $basicSingle.select2();
    $basicMultiple.select2();

    function formatState (state) {
      if (!state.id) {
        return state.text;
      }
      var $state = $(
        '<span>' +
          '<img src="vendor/images/flags/' +
            state.element.value.toLowerCase() +
          '.png" class="img-flag" /> ' +
          state.text +
        '</span>'
      );
      return $state;
    };
  });

</script>

## Container width

Select2 will try to match the width of the original element as closely as possible. Sometimes this isn't perfect, in which case you may manually set the `width` [configuration option](/configuration):

<table class="table table-striped table-bordered">
  <thead>
    <tr>
      <th>Value</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>'element'</code></td>
      <td>
        Uses the computed element width from any applicable CSS rules.
      </td>
    </tr>
    <tr>
      <td><code>'style'</code></td>
      <td>
        Width is determined from the <code>select</code> element's <code>style</code> attribute. If no <code>style</code> attribute is found, null is returned as the width.
      </td>
    </tr>
    <tr>
      <td><code>'resolve'</code></td>
      <td>
        Uses the <code>style</code> attribute value if available, falling back to the computed element width as necessary.
      </td>
    </tr>
    <tr>
      <td><code>'&lt;value&gt;'</code></td>
      <td>
        Valid CSS values can be passed as a string (e.g. <code>width: '80%'</code>).
      </td>
    </tr>
  </tbody>
</table>

### Example

The two Select2 boxes below are styled to `50%` and `75%` width respectively to support responsive design:

<div class="s2-example">
  <p>
    <select class="js-example-responsive js-states" style="width: 50%"></select>
  </p>
  <p>
    <select class="js-example-responsive js-states" multiple="multiple" style="width: 75%"></select>
  </p>
</div>

```
<select class="js-example-responsive" style="width: 50%"></select>
<select class="js-example-responsive" multiple="multiple" style="width: 75%"></select>
```

<pre data-fill-from=".js-code-example-responsive"></pre>

<script type="text/javascript" class="js-code-example-responsive">

$(".js-example-responsive").select2({
    width: 'resolve' // need to override the changed default
});

</script>

>>>> Select2 will do its best to resolve the percent width specified via a CSS class, but it is not always possible. The best way to ensure that Select2 is using a percent based width is to inline the `style` declaration into the tag.

## Themes

Select2 supports custom themes using the `theme` option so you can style Select2 to match the rest of your application.

These examples use the `classic` theme, which matches the old look of Select2.

<div class="s2-example">
  <p>
    <select class="js-example-theme-single js-states form-control">
    </select>
  </p>
  <p>
    <select class="js-example-theme-multiple js-states form-control" multiple="multiple"></select>
  </p>
</div>

<pre data-fill-from=".js-code-example-theme"></pre>

<script type="text/javascript" class="js-code-example-theme">

$(".js-example-theme-single").select2({
  theme: "classic"
});

$(".js-example-theme-multiple").select2({
  theme: "classic"
});

</script>

Various display options of the Select2 component can be changed.  You can access the `<option>` element (or `<optgroup>`) and any attributes on those elements using `.element`.
---
title: Options
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

A traditional `<select>` box contains any number of `<option>` elements.  Each of these is rendered as an option in the dropdown menu.  Select2 preserves this behavior when initialized on a `<select>` element that contains `<option>` elements, converting them into its internal JSON representation:

```
{
  "id": "value attribute" || "option text",
  "text": "label attribute" || "option text",
  "element": HTMLOptionElement
}
```

`<optgroup>` elements will be converted into data objects using the following rules:

```
{
  "text": "label attribute",
  "children": [ option data object, ... ],
  "element": HTMLOptGroupElement
}
```

>>> Options sourced from [other data sources](/data-sources) must conform to this this same internal representation.  See ["The Select2 data format"](/data-sources/formats) for details.

## Dropdown option groups

In HTML, `<option>` elements can be grouped by wrapping them with in an `<optgroup>` element:

```
<select>
  <optgroup label="Group Name">
    <option>Nested option</option>
  </optgroup>
</select>
```

Select2 will automatically pick these up and render them appropriately in the dropdown.

### Hierarchical options

Only a single level of nesting is allowed per the HTML specification. If you nest an `<optgroup>` within another `<optgroup>`, Select2 will not be able to detect the extra level of nesting and errors may be triggered as a result.

Furthermore, `<optgroup>` elements **cannot** be made selectable.  This is a limitation of the HTML specification and is not a limitation that Select2 can overcome.

If you wish to create a true hierarchy of selectable options, use an `<option>` instead of an `<optgroup>` and [change the style with CSS](http://stackoverflow.com/q/30820215/359284#30948247).  Please note that this approach may be considered "less accessible" as it relies on CSS styling, rather than the semantic meaning of `<optgroup>`, to generate the effect.

## Disabling options

Select2 will correctly handle disabled options, both with data coming from a standard select (when the `disabled` attribute is set) and from remote sources, where the object has `disabled: true` set.

<div class="s2-example">
    <select class="js-example-disabled-results form-control">
      <option value="one">First</option>
      <option value="two" disabled="disabled">Second (disabled)</option>
      <option value="three">Third</option>
    </select>
</div>

<pre data-fill-from=".js-code-disabled-option"></pre>

```
<select class="js-example-disabled-results">
  <option value="one">First</option>
  <option value="two" disabled="disabled">Second (disabled)</option>
  <option value="three">Third</option>
</select>
```

<script type="text/javascript" class="js-code-disabled-option">

var $disabledResults = $(".js-example-disabled-results");
$disabledResults.select2();

</script>
---
title: The Select2 data format
taxonomy:
    category: docs
---

Select2 can render programmatically supplied data from an array or remote data source (AJAX) as dropdown options.  In order to accomplish this, Select2 expects a very specific data format.  This format consists of a JSON object containing an array of objects keyed by the `results` key.

```
{
  "results": [
    {
      "id": 1,
      "text": "Option 1"
    },
    {
      "id": 2,
      "text": "Option 2"
    }
  ],
  "pagination": {
    "more": true
  }
}
```

Select2 requires that each object contain an `id` and a `text` property.  Additional parameters passed in with data objects will be included on the data objects that Select2 exposes.

The response object may also contain pagination data, if you would like to use the "infinite scroll" feature.  This should be specified under the `pagination` key.

## Selected and disabled options

You can also supply the `selected` and `disabled` properties for the options in this data structure.  For example:

```
{
  "results": [
    {
      "id": 1,
      "text": "Option 1"
    },
    {
      "id": 2,
      "text": "Option 2",
      "selected": true
    },
    {
      "id": 3,
      "text": "Option 3",
      "disabled": true
    }
  ]
}
```

In this case, Option 2 will be pre-selected, and Option 3 will be [disabled](/options#disabling-options).

## Transforming data into the required format

### Generating `id` properties

Select2 requires that the `id` property is used to uniquely identify the options that are displayed in the results list. If you use a property other than `id` (like `pk`) to uniquely identify an option, you need to map your old property to `id` before passing it to Select2.

If you cannot do this on your server or you are in a situation where the API cannot be changed, you can do this in JavaScript before passing it to Select2:

```
var data = $.map(yourArrayData, function (obj) {
  obj.id = obj.id || obj.pk; // replace pk with your identifier

  return obj;
});
```

### Generating `text` properties

Just like with the `id` property, Select2 requires that the text that should be displayed for an option is stored in the `text` property. You can map this property from any existing property using the following JavaScript:

```
var data = $.map(yourArrayData, function (obj) {
  obj.text = obj.text || obj.name; // replace name with the property used for the text

  return obj;
});
```

## Automatic string casting

Because the `value` attribute on a `<select>` tag must be a string, and Select2 generates the `value` attribute from the `id` property of the data objects, the `id` property on each data object must also be a string.

Select2 will attempt to convert anything that is not a string to a string, which will work for most situations, but it is recommended to explicitly convert your `id`s to strings ahead of time.

Blank `id`s or an `id` with a value of `0` are not permitted.

## Grouped data

When options are to be generated in `<optgroup>` sections, options should be nested under the `children` key of each group object.  The label for the group should be specified as the `text` property on the group's corresponding data object.

```
{
  "results": [
    { 
      "text": "Group 1", 
      "children" : [
        {
            "id": 1,
            "text": "Option 1.1"
        },
        {
            "id": 2,
            "text": "Option 1.2"
        }
      ]
    },
    { 
      "text": "Group 2", 
      "children" : [
        {
            "id": 3,
            "text": "Option 2.1"
        },
        {
            "id": 4,
            "text": "Option 2.2"
        }
      ]
    }
  ],
  "pagination": {
    "more": true
  }
}
```

>>>> Because Select2 generates an `<optgroup>` when creating nested options, only [a single level of nesting is supported](/options#dropdown-option-groups). Any additional levels of nesting is not guaranteed to be displayed properly across all browsers and devices.
---
title: Ajax (remote data)
metadata:
    description: Select2 provides extensive support for populating dropdown items from a remote data source.
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

Select2 comes with AJAX support built in, using jQuery's AJAX methods. In this example, we can search for repositories using GitHub's API:

<select class="js-example-data-ajax form-control"></select>

**In your HTML:**

```
<select class="js-data-example-ajax"></select>
```

**In your Javascript:**

```
$('.js-data-example-ajax').select2({
  ajax: {
    url: 'https://api.github.com/search/repositories',
    dataType: 'json'
    // Additional AJAX parameters go here; see the end of this chapter for the full code of this example
  }
});
```

You can configure how Select2 searches for remote data using the `ajax` option.  Select2 will pass any options in the `ajax` object to jQuery's `$.ajax` function, or the `transport` function you specify.

>>> For **remote data sources only**, Select2 does not create a new `<option>` element until the item has been selected for the first time.  This is done for performance reasons.  Once an `<option>` has been created, it will remain in the DOM even if the selection is later changed.

## Request parameters

Select2 will issue a request to the specified URL when the user opens the control (unless there is a `minimumInputLength` set as a Select2 option), and again every time the user types in the search box.  By default, it will send the following as query string parameters:

- `term` : The current search term in the search box.
- `q`    : Contains the same contents as `term`.
- `_type`: A "request type".  Will usually be `query`, but changes to `query_append` for paginated requests.
- `page` : The current page number to request.  Only sent for paginated (infinite scrolling) searches.

For example, Select2 might issue a request that looks like: `https://api.github.com/search/repositories?term=sel&_type=query&q=sel`.

Sometimes, you may need to add additional query parameters to the request.  You can modify the parameters that are sent with the request by overriding the `ajax.data` option:

```
$('#mySelect2').select2({
  ajax: {
    url: 'https://api.github.com/orgs/select2/repos',
    data: function (params) {
      var query = {
        search: params.term,
        type: 'public'
      }

      // Query parameters will be ?search=[term]&type=public
      return query;
    }
  }
});
```

## Transforming response data

You can use the `ajax.processResults` option to transform the data returned by your API into the format expected by Select2:

```
$('#mySelect2').select2({
  ajax: {
    url: '/example/api',
    processResults: function (data) {
      // Transforms the top-level key of the response object from 'items' to 'results'
      return {
        results: data.items
      };
    }
  }
});
```

>>> Select2 expects results from the remote endpoint to be filtered on the **server side**. See [this comment](https://github.com/select2/select2/issues/2321#issuecomment-42749687) for an explanation of why this implementation choice was made. If server-side filtering is not possible, you may be interested in using Select2's [support for data arrays](/data-sources/arrays) instead.

## Default (pre-selected) values

You may wish to set a pre-selected default value for a Select2 control that receives its data from an AJAX request.

To provide default selections, you may include an `<option>` for each selection that contains the value and text that should be displayed:

```
<select class="js-example-data-ajax form-control">
  <option value="3620194" selected="selected">select2/select2</option>
</select>
```

To achieve this programmatically, you will need to [create and append a new `Option`](/programmatic-control/add-select-clear-items).

## Pagination

Select2 supports pagination ("infinite scrolling") for remote data sources out of the box.  To use this feature, your remote data source must be able to respond to paginated requests (server-side frameworks like [Laravel](https://laravel.com/docs/5.5/pagination) and [UserFrosting](https://learn.userfrosting.com/database/data-sprunjing) have this built-in).

To use pagination, you must tell Select2 to add any necessary pagination parameters to the request by overriding the `ajax.data` setting.  The current page to be retrieved is stored in the `params.page` property.

```
$('#mySelect2').select2({
  ajax: {
    url: 'https://api.github.com/search/repositories',
    data: function (params) {
      var query = {
        search: params.term,
        page: params.page || 1
      }

      // Query parameters will be ?search=[term]&page=[page]
      return query;
    }
  }
});
```

Select2 will expect a `pagination.more` value in the response.  The value of `more` should be `true` or `false`, which tells Select2 whether or not there are more pages of results available for retrieval:

```
{
  "results": [
    {
      "id": 1,
      "text": "Option 1"
    },
    {
      "id": 2,
      "text": "Option 2"
    }
  ],
  "pagination": {
    "more": true
  }
}
```

If your server-side code does not generate the `pagination.more` property in the response, you can use `processResults` to generate this value from other information that is available.  For example, suppose your API returns a `count_filtered` value that tells you how many total (unpaginated) results are available in the data set.  If you know that your paginated API returns 10 results at a time, you can use this along with the value of `count_filtered` to compute the value of `pagination.more`:

```
processResults: function (data, params) {
    params.page = params.page || 1;

    return {
        results: data.results,
        pagination: {
            more: (params.page * 10) < data.count_filtered
        }
    };
}
```

## Rate-limiting requests

You can tell Select2 to wait until the user has finished typing their search term before triggering the AJAX request.  Simply use the `ajax.delay` configuration option to tell Select2 how long to wait after a user has stopped typing before sending the request:

```
$('#mySelect2').select2({
  ajax: {
    delay: 250 // wait 250 milliseconds before triggering the request
  }
});
```

## Dynamic URLs

If there isn't a single url for your search results, or you need to call a function to determine the url to use, you can specify a callback for the `ajax.url` option to generate the url. The current search query will be passed in through the `params` option:

```
$('#mySelect2').select2({
  ajax: {
    url: function (params) {
      return '/some/url/' + params.term;
    }
  }
});
```

## Alternative transport methods

Select2 uses the transport method defined in `ajax.transport` to send requests to your API. By default this transport method is `jQuery.ajax`, but it can be easily overridden:

```
$('#mySelect2').select2({
  ajax: {
    transport: function (params, success, failure) {
      var request = new AjaxRequest(params.url, params);
      request.on('success', success);
      request.on('failure', failure);
    }
  }
});
```

## jQuery `$.ajax` options

All options passed to `ajax` will be directly passed to the `$.ajax` function that executes AJAX requests.

There are a few custom options that Select2 will intercept, allowing you to customize the request as it is being made:

```
ajax: {
  // The number of milliseconds to wait for the user to stop typing before
  // issuing the ajax request.
  delay: 250,
  // You can craft a custom url based on the parameters that are passed into the
  // request. This is useful if you are using a framework which has
  // JavaScript-based functions for generating the urls to make requests to.
  //
  // @param params The object containing the parameters used to generate the
  //   request.
  // @returns The url that the request should be made to.
  url: function (params) {
    return UrlGenerator.Random();
  },
  // You can pass custom data into the request based on the parameters used to
  // make the request. For `GET` requests, the default method, these are the
  // query parameters that are appended to the url. For `POST` requests, this
  // is the form data that will be passed into the request. For other requests,
  // the data returned from here should be customized based on what jQuery and
  // your server are expecting.
  //
  // @param params The object containing the parameters used to generate the
  //   request.
  // @returns Data to be directly passed into the request.
  data: function (params) {
    var queryParameters = {
      q: params.term
    }

    return queryParameters;
  },
  // You can modify the results that are returned from the server, allowing you
  // to make last-minute changes to the data, or find the correct part of the
  // response to pass to Select2. Keep in mind that results should be passed as
  // an array of objects.
  //
  // @param data The data as it is returned directly by jQuery.
  // @returns An object containing the results data as well as any required
  //   metadata that is used by plugins. The object should contain an array of
  //   data objects as the `results` key.
  processResults: function (data) {
    return {
      results: data
    };
  },
  // You can use a custom AJAX transport function if you do not want to use the
  // default one provided by jQuery.
  //
  // @param params The object containing the parameters used to generate the
  //   request.
  // @param success A callback function that takes `data`, the results from the
  //   request.
  // @param failure A callback function that indicates that the request could
  //   not be completed.
  // @returns An object that has an `abort` function that can be called to abort
  //   the request if needed.
  transport: function (params, success, failure) {
    var $request = $.ajax(params);

    $request.then(success);
    $request.fail(failure);

    return $request;
  }
}
```

## Additional examples

This code powers the Github example presented at the beginning of this chapter:

<pre data-fill-from=".js-code-placeholder"></pre>

<script type="text/javascript" class="js-code-placeholder">

$(".js-example-data-ajax").select2({
  ajax: {
    url: "https://api.github.com/search/repositories",
    dataType: 'json',
    delay: 250,
    data: function (params) {
      return {
        q: params.term, // search term
        page: params.page
      };
    },
    processResults: function (data, params) {
      // parse the results into the format expected by Select2
      // since we are using custom formatting functions we do not need to
      // alter the remote JSON data, except to indicate that infinite
      // scrolling can be used
      params.page = params.page || 1;

      return {
        results: data.items,
        pagination: {
          more: (params.page * 30) < data.total_count
        }
      };
    },
    cache: true
  },
  placeholder: 'Search for a repository',
  minimumInputLength: 1,
  templateResult: formatRepo,
  templateSelection: formatRepoSelection
});

function formatRepo (repo) {
  if (repo.loading) {
    return repo.text;
  }

  var $container = $(
    "<div class='select2-result-repository clearfix'>" +
      "<div class='select2-result-repository__avatar'><img src='" + repo.owner.avatar_url + "' /></div>" +
      "<div class='select2-result-repository__meta'>" +
        "<div class='select2-result-repository__title'></div>" +
        "<div class='select2-result-repository__description'></div>" +
        "<div class='select2-result-repository__statistics'>" +
          "<div class='select2-result-repository__forks'><i class='fa fa-flash'></i> </div>" +
          "<div class='select2-result-repository__stargazers'><i class='fa fa-star'></i> </div>" +
          "<div class='select2-result-repository__watchers'><i class='fa fa-eye'></i> </div>" +
        "</div>" +
      "</div>" +
    "</div>"
  );

  $container.find(".select2-result-repository__title").text(repo.full_name);
  $container.find(".select2-result-repository__description").text(repo.description);
  $container.find(".select2-result-repository__forks").append(repo.forks_count + " Forks");
  $container.find(".select2-result-repository__stargazers").append(repo.stargazers_count + " Stars");
  $container.find(".select2-result-repository__watchers").append(repo.watchers_count + " Watchers");

  return $container;
}

function formatRepoSelection (repo) {
  return repo.full_name || repo.text;
}
</script>
---
title: Arrays
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

## Loading array data

You may use the `data` configuration option to load dropdown options from a local array.

You can provide initial selections with array data by providing the option tag for the selected values, similar to how it would be done for a standard select.

<div class="s2-example">
  <p>
    <select class="js-example-data-array form-control"></select>
  </p>
  <p>
    <select class="js-example-data-array-selected form-control">
      <option value="2" selected="selected">duplicate</option>
    </select>
  </p>
</div>

<pre data-fill-from=".js-code-placeholder"></pre>

<script type="text/javascript" class="js-code-placeholder">

var data = [
    {
        id: 0,
        text: 'enhancement'
    },
    {
        id: 1,
        text: 'bug'
    },
    {
        id: 2,
        text: 'duplicate'
    },
    {
        id: 3,
        text: 'invalid'
    },
    {
        id: 4,
        text: 'wontfix'
    }
];

$(".js-example-data-array").select2({
  data: data
})

$(".js-example-data-array-selected").select2({
  data: data
})
</script>

Unlike the case of items supplied by [AJAX data sources](/data-sources/ajax), items supplied as an array will be immediately rendered as `<option>` elements in the target `<select>` control.

## Backwards compatibility with the `tags` option

In Select2 v3.5, this option was called `tags`.  However in version 4.0, `tags` now handles the [tagging feature](/tagging).

For backwards compatibility the `tags` option can still accept an array of objects, in which case they will be handled in the same manner as the `data` option.
  ---
title: Data sources
taxonomy:
    category: docs
---

# Data sources

In addition to handling `<option>` elements that explicitly appear in your markup, Select2 can also retrieve the results from other data sources such as a remote JSON API or a local Javascript array.
---
title: Dropdown
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

This chapter covers the appearance and behavior of the list of results in the dropdown menu.

## Templating

By default, Select2 will display the `text` property of each data object within the list of results.  The appearance of search results in the dropdown can be customized by using the `templateResult` option:

<div class="s2-example">
    <select class="js-example-templating js-states form-control"></select>
</div>

<pre data-fill-from=".js-code-example-templating"></pre>

<script type="text/javascript" class="js-code-example-templating">

function formatState (state) {
  if (!state.id) {
    return state.text;
  }
  var baseUrl = "{{ url('user://pages/images/flags') }}";
  var $state = $(
    '<span><img src="' + baseUrl + '/' + state.element.value.toLowerCase() + '.png" class="img-flag" /> ' + state.text + '</span>'
  );
  return $state;
};

$(".js-example-templating").select2({
  templateResult: formatState
});

</script>

The `templateResult` function should return a string containing the text to be displayed, or an object (such as a jQuery object) that contains the data that should be displayed.  It can also return `null`, which will prevent the option from being displayed in the results list.

>>> You may find it helpful to use a client-side templating engine like [Handlebars](http://handlebarsjs.com/) to define your templates.

### Built-in escaping

By default, strings returned by `templateResult` are assumed to **contain only text** and will be passed through the `escapeMarkup` function, which strips any HTML markup.

If you need to render HTML with your result template, you must wrap your rendered result in a jQuery object. In this case, the result will be passed [directly to `jQuery.fn.append`](https://api.jquery.com/append/) and will be handled directly by jQuery.  Any markup, such as HTML, will not be escaped and it is up to you to escape any malicious input provided by users.

>>> **Anything rendered in the results is templated.** This includes results such as the "Searching..." and "Loading more..." text which will periodically be displayed, which allows you to add more advanced formatting to these automatically generated options.  You must ensure that your templating functions can support them.

## Automatic selection

Select2 can be configured to automatically select the currently highlighted result when the dropdown is closed by using the `selectOnClose` option:

```
$('#mySelect2').select2({
  selectOnClose: true
});
```

## Forcing the dropdown to remain open after selection

Select2 will automatically close the dropdown when an element is selected, similar to what is done with a normal select box.  You may use the `closeOnSelect` option to prevent the dropdown from closing when a result is selected:

```
$('#mySelect2').select2({
  closeOnSelect: false
});
```

Note that this option is only applicable to multi-select controls.

>>> If the [`CloseOnSelect` decorator](/advanced/default-adapters/dropdown#closeonselect) is not used (or `closeOnSelect` is set to <code>false</code>), the dropdown will not automatically close when a result is selected.  The dropdown will also never close if the <kbd>ctrl</kbd> key is held down when the result is selected.

## Dropdown placement

>>>>> Attention [Harvest Chosen](https://harvesthq.github.io/chosen/) migrators!  If you are migrating to Select2 from Chosen, this option will cause Select2 to position the dropdown in a similar way.

By default, Select2 will attach the dropdown to the end of the body and will absolutely position it to appear above or below the selection container.

Select2 will display the dropdown above the container if there is not enough space below the container, but there is enough space above it.

The `dropdownParent` option allows you to pick an alternative element for the dropdown to be appended to:

```
$('#mySelect2').select2({
  dropdownParent: $('#myModal')
});
```

This is useful when attempting to render Select2 correctly inside of modals and other small containers.  If you're having trouble using the search box inside a Bootstrap modal, for example, trying setting the `dropdownParent` option to the modal element.

If you are rendering a Select2 inside of a modal (Bootstrap 3.x) that has not yet been rendered or opened, you may need to bind to the `shown.bs.modal` event:

```
$('body').on('shown.bs.modal', '.modal', function() {
  $(this).find('select').each(function() {
    var dropdownParent = $(document.body);
    if ($(this).parents('.modal.in:first').length !== 0)
      dropdownParent = $(this).parents('.modal.in:first');
    $(this).select2({
      dropdownParent: dropdownParent
      // ...
    });
  });
});
```

If you run into positioning issues while using the default `body` attachment, you may find it helpful to use your browser console to inspect the values of:

- `document.body.style.position`
- `$(document.body).offset()`

See [this issue](https://github.com/select2/select2/issues/3970#issuecomment-160496724).

>>>> `dropdownParent` will cause DOM events to be raised outside of the standard Select2 DOM container. This can cause issues with third-party components such as modals.
---
title: Selections
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

When an option is selected from the dropdown menu, Select2 will display the selected value in the container box.  By default, it will display the `text` property of Select2's [internal representation of the selected option](/options).

## Templating

The appearance of selected results can be customized by using the `templateSelection` configuration option.  This takes a callback that transforms the selection data object into a string representation or jQuery object:

<div class="s2-example">
    <select class="js-example-templating js-states form-control"></select>
</div>

<pre data-fill-from=".js-code-example-templating"></pre>

<script type="text/javascript" class="js-code-example-templating">

function formatState (state) {
  if (!state.id) {
    return state.text;
  }

  var baseUrl = "{{ url('user://pages/images/flags') }}";
  var $state = $(
    '<span><img class="img-flag" /> <span></span></span>'
  );

  // Use .text() instead of HTML string concatenation to avoid script injection issues
  $state.find("span").text(state.text);
  $state.find("img").attr("src", baseUrl + "/" + state.element.value.toLowerCase() + ".png");

  return $state;
};

$(".js-example-templating").select2({
  templateSelection: formatState
});

</script>

>>> You may find it helpful to use a client-side templating engine like [Handlebars](http://handlebarsjs.com/) to define your templates.

### Built-in escaping

By default, strings returned by `templateSelection` are assumed to **contain only text** and will be passed through the `escapeMarkup` function, which strips any HTML markup.

If you need to render HTML with your selection template, you must wrap your rendered selection in a jQuery object. In this case, the selection will be passed [directly to `jQuery.fn.append`](https://api.jquery.com/append/) and will be handled directly by jQuery.  Any markup, such as HTML, will not be escaped and it is up to you to escape any malicious input provided by users.

>>>> Anything rendered as a selection is templated.  This includes placeholders and pre-existing selections that are displayed, so you must ensure that your templating functions can support them.

## Limiting the number of selections

Select2 multi-value select boxes can set restrictions regarding the maximum number of options that can be selected. The select below is declared with the `multiple` attribute with `maximumSelectionLength` in the select2 options.

<div class="s2-example">
    <p>
      <select class="js-example-basic-multiple-limit js-states form-control" multiple="multiple"></select>
    </p>
</div>

<pre data-fill-from=".js-code-placeholder"></pre>

<script type="text/javascript" class="js-code-placeholder">

$(".js-example-basic-multiple-limit").select2({
  maximumSelectionLength: 2
});

</script>

## Clearable selections

When set to `true`, causes a clear button ("x" icon) to appear on the select box when a value is selected. Clicking the clear button will clear the selected value, effectively resetting the select box back to its placeholder value.

```
$('select').select2({
  placeholder: 'This is my placeholder',
  allowClear: true
});
```
---
title: Dynamic option creation
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

In addition to a prepopulated menu of options, Select2 can dynamically create new options from text input by the user in the search box.  This feature is called "tagging".  To enable tagging, set the `tags` option to `true`:

<div class="s2-example">
  <p>
    <select class="js-example-tags form-control">
      <option selected="selected">orange</option>
      <option>white</option>
      <option>purple</option>
    </select>
  </p>
</div>

```
<select class="form-control">
  <option selected="selected">orange</option>
  <option>white</option>
  <option>purple</option>
</select>

$(".js-example-tags").select2({
  tags: true
});
```

Note that when tagging is enabled the user can select from the pre-existing options or create a new option by picking the first choice, which is what the user has typed into the search box so far.

## Tagging with multi-value select boxes

Tagging can also be used in multi-value select boxes. In the example below, we set the `multiple="multiple"` attribute on a Select2 control that also has `tags: true` enabled:
  
<div class="s2-example">
  <p>
    <select class="js-example-tags form-control" multiple="multiple">
      <option selected="selected">orange</option>
      <option>white</option>
      <option selected="selected">purple</option>
    </select>
  </p>
</div>

```
<select class="form-control" multiple="multiple">
  <option selected="selected">orange</option>
  <option>white</option>
  <option selected="selected">purple</option>
</select>
```

<script type="text/javascript">

$(".js-example-tags").select2({
  tags: true
});

</script>

Try entering a value that isn't listed in the dropdown - you'll be able to add it as a new option!

## Automatic tokenization into tags

Select2 supports ability to add choices automatically as the user is typing into the search field. Try typing in the search field below and entering a space or a comma.

The separators that should be used when tokenizing can be specified using the `tokenSeparators` options.

<div class="s2-example">
<p>
  <select class="js-example-tokenizer form-control" multiple="multiple">
    <option>red</option>
    <option>blue</option>
    <option>green</option>
  </select>
</p>
</div>

<pre data-fill-from=".js-code-example-tokenizer"></pre>

<script type="text/javascript" class="js-code-example-tokenizer">

$(".js-example-tokenizer").select2({
    tags: true,
    tokenSeparators: [',', ' ']
})

</script>

## Customizing tag creation

### Tag properties

You may add extra properties to newly created tags by defining a `createTag` callback:

```
$('select').select2({
  createTag: function (params) {
    var term = $.trim(params.term);

    if (term === '') {
      return null;
    }

    return {
      id: term,
      text: term,
      newTag: true // add additional parameters
    }
  }
});
```

### Constraining tag creation

You may control when Select2 will allow the user to create a new tag, by adding some logic to `createTag` to return `null` if an invalid value is entered:

```
$('select').select2({
  createTag: function (params) {
    // Don't offset to create a tag if there is no @ symbol
    if (params.term.indexOf('@') === -1) {
      // Return null to disable tag creation
      return null;
    }

    return {
      id: params.term,
      text: params.term
    }
  }
});
```

## Customizing tag placement in the dropdown

You may control the placement of the newly created option by defining a `insertTag` callback:

```
$('select').select2({
  insertTag: function (data, tag) {
    // Insert the tag at the end of the results
    data.push(tag);
  }
});
```
---
title: Placeholders
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

Select2 supports displaying a placeholder value using the `placeholder` configuration option. The placeholder value will be displayed until a selection is made.

## Text placeholders

The most common situation is to use a string of text as your placeholder value.

### Single select placeholders

<div class="s2-example">
  <p>
    <select class="js-example-placeholder-single js-states form-control">
      <option></option>
    </select>
  </p>
</div>

```html
<select class="js-example-placeholder-single js-states form-control">
  <option></option>
</select>
```

<pre data-fill-from="#example-placeholder-single-select"></pre>

<script type="text/javascript" id="example-placeholder-single-select" class="js-code-placeholder">
$(".js-example-placeholder-single").select2({
    placeholder: "Select a state",
    allowClear: true
});
</script>

>>> **For single selects only**, in order for the placeholder value to appear, you must have a blank `<option>` as the first option in your `<select>` control.  This is because the browser tries to select the first option by default. If your first option were non-empty, the browser would display this instead of the placeholder.

### Multi-select placeholders

For multi-selects, you must **not** have an empty `<option>` element:

<select class="js-example-placeholder-multiple js-states form-control" multiple="multiple"></select>

```html
<select class="js-example-placeholder-multiple js-states form-control" multiple="multiple"></select>
```

<pre data-fill-from="#example-placeholder-multi-select"></pre>

<script type="text/javascript" id="example-placeholder-multi-select" class="js-code-placeholder">
$(".js-example-placeholder-multiple").select2({
    placeholder: "Select a state"
});
</script>

>>> Select2 uses the `placeholder` attribute on multiple select boxes, which requires IE 10+. You can support it in older versions with [the Placeholders.js polyfill](https://github.com/jamesallardice/Placeholders.js).

## Default selection placeholders

Alternatively, the value of the `placeholder` option can be a data object representing a default selection (`<option>`). In this case the `id` of the data object should match the `value` of the corresponding default selection.

```
$('select').select2({
  placeholder: {
    id: '-1', // the value of the option
    text: 'Select an option'
  }
});
```

This is useful, for example, when you are using a framework that creates its own placeholder option.

## Using placeholders with AJAX

Select2 supports placeholders for all configurations, including AJAX. You will still need to add in the empty `<option>` if you are using a single select.

## Customizing placeholder appearance

When using Select2 **in single-selection mode**, the placeholder option will be passed through the `templateSelection` callback if specified. You can use some additional logic in this callback to check the `id` property and apply an alternative transformation to your placeholder option:

```
$('select').select2({
  templateSelection: function (data) {
    if (data.id === '') { // adjust for custom placeholder values
      return 'Custom styled placeholder text';
    }

    return data.text;
  }
});
```

>>>>> **When multiple selections are allowed**, the placeholder will be displayed using the `placeholder` attribute on the search box. You can customize the display of this placeholder using CSS, as explained in the following Stack Overflow answer: [Change an input's HTML5 placeholder color with CSS](http://stackoverflow.com/q/2610497/359284).

## Placeholders in legacy Internet Explorer versions

Select2 uses the native `placeholder` attribute on input boxes for the multiple select, and that attribute is not supported in older versions of Internet Explorer. You need to include [Placeholders.js](https://github.com/jamesallardice/Placeholders.js) on your page, or use the full build, in order to add `placeholder` attribute support to input boxes.
---
title: Search
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

A search box is added to the top of the dropdown automatically for select boxes where only a single option can be selected. The behavior and appearance of the search box can be easily customized with Select2.

## Customizing how results are matched

When users filter down the results by entering search terms into the search box, Select2 uses an internal "matcher" to match search terms to results. You may customize the way that Select2 matches search terms by specifying a callback for the `matcher` configuration option.

Select2 will pass each option as represented by its [internal representation](/options) into this callback to determine if it should be displayed:

```
function matchCustom(params, data) {
    // If there are no search terms, return all of the data
    if ($.trim(params.term) === '') {
      return data;
    }

    // Do not display the item if there is no 'text' property
    if (typeof data.text === 'undefined') {
      return null;
    }

    // `params.term` should be the term that is used for searching
    // `data.text` is the text that is displayed for the data object
    if (data.text.indexOf(params.term) > -1) {
      var modifiedData = $.extend({}, data, true);
      modifiedData.text += ' (matched)';
   
      // You can return modified objects from here
      // This includes matching the `children` how you want in nested data sets
      return modifiedData;
    }
   
    // Return `null` if the term should not be displayed
    return null;
}
    
$(".js-example-matcher").select2({
  matcher: matchCustom
});
```

>>>> `matcher` only works with **locally supplied data** (e.g., via an [array](/data-sources/arrays)!  When a remote data set is used, Select2 expects that the returned results have already been filtered on the server side.

### Matching grouped options

Only first-level objects will be passed in to the `matcher` callback.  If you are working with nested data, you must iterate through the `children` array and match them individually.  This allows for more advanced matching when working with nested objects, allowing you to handle them however you want.

This example matches results only if the term appears in the beginning of the string:

<div class="s2-example">
    <select class="js-example-matcher-start js-states form-control"></select>
</div>

<pre data-fill-from=".js-code-example-matcher"></pre>

<script type="text/javascript" class="js-code-example-matcher">

function matchStart(params, data) {
  // If there are no search terms, return all of the data
  if ($.trim(params.term) === '') {
    return data;
  }

  // Skip if there is no 'children' property
  if (typeof data.children === 'undefined') {
    return null;
  }

  // `data.children` contains the actual options that we are matching against
  var filteredChildren = [];
  $.each(data.children, function (idx, child) {
    if (child.text.toUpperCase().indexOf(params.term.toUpperCase()) == 0) {
      filteredChildren.push(child);
    }
  });

  // If we matched any of the timezone group's children, then set the matched children on the group
  // and return the group object
  if (filteredChildren.length) {
    var modifiedData = $.extend({}, data, true);
    modifiedData.children = filteredChildren;

    // You can return modified objects from here
    // This includes matching the `children` how you want in nested data sets
    return modifiedData;
  }

  // Return `null` if the term should not be displayed
  return null;
}

$(".js-example-matcher-start").select2({
  matcher: matchStart
});

</script>

>>> A [compatibility module](/upgrading/migrating-from-35#wrapper-for-old-style-matcher-callbacks) exists for using v3-style matcher callbacks.

## Minimum search term length

Sometimes when working with large data sets, it is more efficient to start filtering the results only when the search term is a certain length. This is very common when working with remote data sets, as it allows for only significant search terms to be used.

You may set a minimum search term length  by using the `minimumInputLength` option:

```
$('select').select2({
  minimumInputLength: 3 // only start searching when the user has input 3 or more characters
});
```

## Maximum search term length

In some cases, search terms need to be limited to a certain range. Select2 allows you to limit the length of the search term such that it does not exceed a certain length.

You may limit the maximum length of search terms by using the `maximumInputLength` option:

```
$('select').select2({
    maximumInputLength: 20 // only allow terms up to 20 characters long
});
```

## Limiting display of the search box to large result sets

The `minimumResultsForSearch` option determines the minimum number of results required in the initial population of the dropdown to display the search box.

This option is useful for cases where local data is used with a small result set, and the search box would simply be a waste of screen real estate. Set this value to -1 to permanently hide the search box.

```
$('select').select2({
    minimumResultsForSearch: 20 // at least 20 results must be displayed
});
```

## Hiding the search box

### Single select

For single selects, Select2 allows you to hide the search box using the `minimumResultsForSearch` configuration option. In this example, we use the value `Infinity` to tell Select2 to never display the search box.

<div class="s2-example">
    <select id="js-example-basic-hide-search" class="js-states form-control"></select>
</div>

<pre data-fill-from="#js-code-example-basic-hide-search"></pre>

<script type="text/javascript" id="js-code-example-basic-hide-search">

$("#js-example-basic-hide-search").select2({
    minimumResultsForSearch: Infinity
});

</script>

### Multi-select

For multi-select boxes, there is no distinct search control. So, to disable search for multi-select boxes, you will need to set the `disabled` property to true whenever the dropdown is opened or closed:

<div class="s2-example">
    <select id="js-example-basic-hide-search-multi" class="js-states form-control" multiple="multiple"></select>
</div>

<pre data-fill-from="#js-code-example-basic-hide-search-multi"></pre>

<script type="text/javascript" id="js-code-example-basic-hide-search-multi">

$('#js-example-basic-hide-search-multi').select2();

$('#js-example-basic-hide-search-multi').on('select2:opening select2:closing', function( event ) {
    var $searchfield = $(this).parent().find('.select2-search__field');
    $searchfield.prop('disabled', true);
});
</script>

See [this issue](https://github.com/select2/select2/issues/4797) for the source of this solution.
---
title: Add, select, or clear items
metadata:
    description: Programmatically adding, selecting, and clearing options in a Select2 control.
taxonomy:
    category: docs
---

## Creating new options in the dropdown

New options can be added to a Select2 control programmatically by creating a new [Javascript `Option` object](https://developer.mozilla.org/en-US/docs/Web/API/HTMLOptionElement/Option) and appending it to the control:

```
var data = {
    id: 1,
    text: 'Barn owl'
};

var newOption = new Option(data.text, data.id, false, false);
$('#mySelect2').append(newOption).trigger('change');
```

The third parameter of `new Option(...)` determines whether the item is "default selected"; i.e. it sets the `selected` attribute for the new option.  The fourth parameter sets the options actual selected state - if set to `true`, the new option will be selected by default.

### Create if not exists

You can use `.find` to select the option if it already exists, and create it otherwise:

```
// Set the value, creating a new option if necessary
if ($('#mySelect2').find("option[value='" + data.id + "']").length) {
    $('#mySelect2').val(data.id).trigger('change');
} else { 
    // Create a DOM Option and pre-select by default
    var newOption = new Option(data.text, data.id, true, true);
    // Append it to the select
    $('#mySelect2').append(newOption).trigger('change');
} 
```

## Selecting options

To programmatically select an option/item for a Select2 control, use the jQuery `.val()` method:

```
$('#mySelect2').val('1'); // Select the option with a value of '1'
$('#mySelect2').trigger('change'); // Notify any JS components that the value changed
```

You can also pass an array to `val` make multiple selections:

```
$('#mySelect2').val(['1', '2']);
$('#mySelect2').trigger('change'); // Notify any JS components that the value changed
```

Select2 will listen for the `change` event on the `<select>` element that it is attached to. When you make any external changes that need to be reflected in Select2 (such as changing the value), you should trigger this event.

### Preselecting options in an remotely-sourced (AJAX) Select2 

For Select2 controls that receive their data from an [AJAX source](/data-sources/ajax), using `.val()` will not work.  The options won't exist yet, because the AJAX request is not fired until the control is opened and/or the user begins searching.  This is further complicated by server-side filtering and pagination - there is no guarantee when a particular item will actually be loaded into the Select2 control!

The best way to deal with this, therefore, is to simply add the preselected item as a new option.  For remotely sourced data, this will probably involve creating a new API endpoint in your server-side application that can retrieve individual items:

```
// Set up the Select2 control
$('#mySelect2').select2({
    ajax: {
        url: '/api/students'
    }
});

// Fetch the preselected item, and add to the control
var studentSelect = $('#mySelect2');
$.ajax({
    type: 'GET',
    url: '/api/students/s/' + studentId
}).then(function (data) {
    // create the option and append to Select2
    var option = new Option(data.full_name, data.id, true, true);
    studentSelect.append(option).trigger('change');

    // manually trigger the `select2:select` event
    studentSelect.trigger({
        type: 'select2:select',
        params: {
            data: data
        }
    });
});
```

Notice that we manually trigger the `select2:select` event and pass along the entire `data` object.  This allows other handlers to [access additional properties of the selected item](/programmatic-control/events#triggering-events).

## Clearing selections

You may clear all current selections in a Select2 control by setting the value of the control to `null`:

```
$('#mySelect2').val(null).trigger('change');
```
---
title: Retrieving selections
metadata:
    description: There are two ways to programmatically access the current selection data: using `.select2('data')`, or by using a jQuery selector.
taxonomy:
    category: docs
---

There are two ways to programmatically access the current selection data: using `.select2('data')`, or by using a jQuery selector.

## Using the `data` method

Calling `select2('data')` will return a JavaScript array of objects representing the current selection. Each object will contain all of the properties/values that were in the source data objects passed through `processResults` and `templateResult` callbacks.

```
$('#mySelect2').select2('data');
```

## Using a jQuery selector

Selected items can also be accessed via the `:selected` jQuery selector:

```
$('#mySelect2').find(':selected');
```

It is possible to extend the `<option>` elements representing the current selection(s) with HTML `data-*` attributes to contain arbitrary data from the source data objects:

```
$('#mySelect2').select2({
  // ...
  templateSelection: function (data, container) {
    // Add custom attributes to the <option> tag for the selected option
    $(data.element).attr('data-custom-attribute', data.customValue);
    return data.text;
  }
});

// Retrieve custom attribute value of the first selected element
$('#mySelect2').find(':selected').data('custom-attribute');
```

>>>> Do not rely on the `selected` attribute of `<option>` elements to determine the currently selected item(s).  Select2 does not add the `selected` attribute when an element is created from a remotely-sourced option.  See [this issue](https://github.com/select2/select2/issues/3366#issuecomment-102566500) for more information.
---
title: Methods
metadata:
    description: Select2 has several built-in methods that allow programmatic control of the component. 
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

Select2 has several built-in methods that allow programmatic control of the component.  

## Opening the dropdown

Methods handled directly by Select2 can be invoked by passing the name of the method to `.select2(...)`.

The `open` method will cause the dropdown menu to open, displaying the selectable options:

```
$('#mySelect2').select2('open');
```

## Closing the dropdown

The `close` method will cause the dropdown menu to close, hiding the selectable options:

```
$('#mySelect2').select2('close');
```

## Checking if the plugin is initialized

To test whether Select2 has been initialized on a particular DOM element, you can check for the `select2-hidden-accessible` class:

```
if ($('#mySelect2').hasClass("select2-hidden-accessible")) {
    // Select2 has been initialized
}
```

See [this Stack Overflow answer](https://stackoverflow.com/a/29854133/2970321)).

## Destroying the Select2 control

The `destroy` method will remove the Select2 widget from the target element.  It will revert back to a standard `select` control:
  
```
$('#mySelect2').select2('destroy');
```

### Event unbinding

When you destroy a Select2 control, Select2 will only unbind the events that were automatically bound by the plugin.  Any events that you bind in your own code, **including any [Select2 events](/programmatic-control/events) that you explicitly bind,** will need to be unbound manually using the `.off` jQuery method:

```
// Set up a Select2 control
$('#example').select2();

// Bind an event
$('#example').on('select2:select', function (e) { 
    console.log('select event');
});

// Destroy Select2
$('#example').select2('destroy');

// Unbind the event
$('#example').off('select2:select');
```

## Examples

<div class="s2-example">

    <label for="select2-single">Single select</label>
    
    <button class="js-programmatic-set-val button" aria-label="Set Select2 option">
      Set "California"
    </button>
    
    <button class="js-programmatic-open button">
      Open
    </button>
    
    <button class="js-programmatic-close button">
      Close
    </button>
    
    <button class="js-programmatic-destroy button">
      Destroy
    </button>
    
    <button class="js-programmatic-init button">
      Re-initialize
    </button>
    <p>
      <select id="select2-single" class="js-example-programmatic js-states form-control"></select>
    </p>
    
    <label for="select2-multi">Multiple select</label>

    <button type="button" class="js-programmatic-multi-set-val button" aria-label="Programmatically set Select2 options">
      Set to California and Alabama
    </button>
    
    <button type="button" class="js-programmatic-multi-clear button" aria-label="Programmatically clear Select2 options">
      Clear
    </button>

    <p>
      <select id="select2-multi" class="js-example-programmatic-multi js-states form-control" multiple="multiple"></select>
    </p>

</div>

<pre data-fill-from=".js-code-programmatic"></pre>

<script type="text/javascript" class="js-code-programmatic">

var $example = $(".js-example-programmatic").select2();
var $exampleMulti = $(".js-example-programmatic-multi").select2();

$(".js-programmatic-set-val").on("click", function () {
    $example.val("CA").trigger("change");
});

$(".js-programmatic-open").on("click", function () {
    $example.select2("open");
});

$(".js-programmatic-close").on("click", function () {
    $example.select2("close");
});

$(".js-programmatic-init").on("click", function () {
    $example.select2();
});

$(".js-programmatic-destroy").on("click", function () {
    $example.select2("destroy");
});

$(".js-programmatic-multi-set-val").on("click", function () {
    $exampleMulti.val(["CA", "AL"]).trigger("change");
});

$(".js-programmatic-multi-clear").on("click", function () {
    $exampleMulti.val(null).trigger("change");
});

</script>
---
title: Events
metadata:
    description: Listening to Select2's built-in events, and manually triggering events on the Select2 component.
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

Select2 will trigger a few different events when different actions are taken using the component, allowing you to add custom hooks and perform actions.  You may also manually trigger these events on a Select2 control using `.trigger`.

| Event | Description |
| ----- | ----------- |
| `change` | Triggered whenever an option is selected or removed. |
| `change.select2` | Scoped version of `change`.  See [below](#limiting-the-scope-of-the-change-event) for more details. |
| `select2:closing` | Triggered before the dropdown is closed. This event can be prevented. |
| `select2:close` | Triggered whenever the dropdown is closed. `select2:closing` is fired before this and can be prevented. |
| `select2:opening` | Triggered before the dropdown is opened. This event can be prevented. |
| `select2:open` | Triggered whenever the dropdown is opened. `select2:opening` is fired before this and can be prevented. |
| `select2:selecting` | Triggered before a result is selected. This event can be prevented. |
| `select2:select` | Triggered whenever a result is selected. `select2:selecting` is fired before this and can be prevented. |
| `select2:unselecting` | Triggered before a selection is removed. This event can be prevented. |
| `select2:unselect` | Triggered whenever a selection is removed. `select2:unselecting` is fired before this and can be prevented. |
| `select2:clearing` | Triggered before all selections are cleared. This event can be prevented. |
| `select2:clear` | Triggered whenever all selections are cleared. `select2:clearing` is fired before this and can be prevented. |

## Listening for events

All public events are relayed using the jQuery event system, and they are triggered on the `<select>` element that Select2 is attached to. You can attach to them using the [`.on` method](https://api.jquery.com/on/) provided by jQuery:

```
$('#mySelect2').on('select2:select', function (e) {
  // Do something
});
```

## Event data

When `select2:select` is triggered, data from the selection can be accessed via the `params.data` property:

```
$('#mySelect2').on('select2:select', function (e) {
    var data = e.params.data;
    console.log(data);
});
```

`e.params.data` will return an object containing the selection properties.  Any additional data for the selection that was provided in the [data source](/data-sources/formats) will be included in this object as well.  For example:

```
{
  "id": 1,
  "text": "Tyto alba",
  "genus": "Tyto",
  "species": "alba"
}
```

## Triggering events

You can manually trigger events on a Select2 control using the jQuery [`trigger`](http://api.jquery.com/trigger/) method.  However, if you want to pass some data to any handlers for the event, you need to construct a new [jQuery `Event` object](http://api.jquery.com/category/events/event-object/) and trigger on that:

```
var data = {
  "id": 1,
  "text": "Tyto alba",
  "genus": "Tyto",
  "species": "alba"
};

$('#mySelect2').trigger({
    type: 'select2:select',
    params: {
        data: data
    }
});
```

### Limiting the scope of the `change` event

It's common for other components to be listening to the `change` event, or for custom event handlers to be attached that may have side effects.  To limit the scope to **only** notify Select2 of the change, use the `.select2` event namespace:

```
$('#mySelect2').val('US'); // Change the value or make some change to the internal state
$('#mySelect2').trigger('change.select2'); // Notify only Select2 of changes
```

## Examples

<div class="s2-example">
  <p>
    <select class="js-states js-example-events form-control"></select>
  </p>
  <p>
    <select class="js-states js-example-events form-control" multiple="multiple"></select>
  </p>
</div>

<div class="s2-event-log">
  <ul class="js-event-log"></ul>
</div>

<pre data-fill-from=".js-code-events"></pre>

<script type="text/javascript" class="js-code-events">
var $eventLog = $(".js-event-log");
var $eventSelect = $(".js-example-events");

$eventSelect.select2();

$eventSelect.on("select2:open", function (e) { log("select2:open", e); });
$eventSelect.on("select2:close", function (e) { log("select2:close", e); });
$eventSelect.on("select2:select", function (e) { log("select2:select", e); });
$eventSelect.on("select2:unselect", function (e) { log("select2:unselect", e); });

$eventSelect.on("change", function (e) { log("change"); });

function log (name, evt) {
  if (!evt) {
    var args = "{}";
  } else {
    var args = JSON.stringify(evt.params, function (key, value) {
      if (value && value.nodeName) return "[DOM node]";
      if (value instanceof $.Event) return "[$.Event]";
      return value;
    });
  }
  var $e = $("<li>" + name + " -> " + args + "</li>");
  $eventLog.append($e);
  $e.animate({ opacity: 1 }, 10000, 'linear', function () {
    $e.animate({ opacity: 0 }, 2000, 'linear', function () {
      $e.remove();
    });
  });
}
</script>

## Preventing events

See [https://stackoverflow.com/a/26706695/2970321](https://stackoverflow.com/a/26706695/2970321).

## Internal Select2 events

Select2 has an [internal event system](/advanced/default-adapters/selection#eventrelay) that works independently of the DOM event system, allowing adapters to communicate with each other. This internal event system is only accessible from plugins and adapters that are connected to Select2 - **not** through the jQuery event system.

You can find more information on the public events triggered by individual adapters in the [advanced chapter](/advanced).
---
title: Programmatic control
metadata:
    description: Programmatically adding and selecting items, getting the current selections, manipulating the control, and working with Select2 events.
taxonomy:
    category: docs
---

# Programmatic control
---
title: Internationalization
taxonomy:
    category: docs
process:
    twig: true
never_cache_twig: true
---

{% do assets.addJs('https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/i18n/es.js', 90) %}

## Message translations

When necessary, Select2 displays certain messages to the user.  For example, a message will appear when no search results were found or more characters need to be entered in order for a search to be made. These messages have been translated into many languages by contributors to Select2, but you can also provide your own translations.

### Language files

Select2 can load message translations for different languages from language files.  When using translations provided by Select2, you must make sure to include the translation file in your page after Select2.

When a string is passed in as the language, Select2 will try to resolve it into a language file. This allows you to specify your own language files, which must be defined as an AMD module. If the language file cannot be found, Select2 will assume it is is one of Select2's built-in languages, and it will try to load the translations for that language instead.

<div class="s2-example">
    <p>
      <select class="js-example-language js-states form-control">
      </select>
    </p>
</div>

```
$(".js-example-language").select2({
  language: "es"
});
```

<script type="text/javascript">
    $(".js-example-language").select2({
      language: "es"
    });
</script>

The language does not have to be defined when Select2 is being initialized, but instead can be defined in the `[lang]` attribute of any parent elements as `[lang="es"]`.

### Translation objects

You may alternatively provide your own custom messages to be displayed by providing an object similar to the one below:

```
language: {
  // You can find all of the options in the language files provided in the
  // build. They all must be functions that return the string that should be
  // displayed.
  inputTooShort: function () {
    return "You must enter more characters...";
  }
}
```

>>> Translations are handled by the `select2/translation` module.

## RTL support

Select2 will work on RTL websites if the `dir` attribute is set on the `<select>` or any parents of it. You can also initialize Select2 with the `dir: "rtl"` configuration option.

<div class="s2-example">
    <p>
      <select class="js-example-rtl js-states form-control" dir="rtl"></select>
    </p>
</div>

```
$(".js-example-rtl").select2({
  dir: "rtl"
});
```

<script type="text/javascript">
    $(".js-example-rtl").select2({
      dir: "rtl"
    });
</script>

## Transliteration support (diacritics)

Select2's default matcher will transliterate diacritic-modified letters into their ASCII counterparts, making it easier for users to filter results in international selects. Type "aero" into the select below.

<div class="s2-example">
  <p>
    <select class="js-example-diacritics form-control">
      <option>Aeróbics</option>
      <option>Aeróbics en Agua</option>
      <option>Aerografía</option>
      <option>Aeromodelaje</option>
      <option>Águilas</option>
      <option>Ajedrez</option>
      <option>Ala Delta</option>
      <option>Álbumes de Música</option>
      <option>Alusivos</option>
      <option>Análisis de Escritura a Mano</option>
    </select>
  </p>
</div>

```
$(".js-example-diacritics").select2();
```

<script type="text/javascript">
    $(".js-example-diacritics").select2();
</script>
---
title: Adapters and Decorators
taxonomy:
    category: docs
---

Starting in version 4.0, Select2 uses the [Adapter pattern](https://en.wikipedia.org/wiki/Adapter_pattern) as a powerful means of extending its features and behavior.

Most of the built-in features, such as those described in the previous chapters, are implemented via one of the [built-in adapters](/advanced/default-adapters).  You may further extend the functionality of Select2 by implementing your own adapters.

## Adapter interfaces

All custom adapters must implement the methods described by the `Adapter` interface.

In addition, adapters that override the default `selectionAdapter` and `dataAdapter` behavior must implement the additional methods described by the corresponding `SelectionAdapter` and `DataAdapter` interfaces.

### `Adapter`

All adapters must implement the `Adapter` interface, which Select2 uses to render DOM elements for the adapter and bind any internal events:

```
// The basic HTML that should be rendered by Select2. A jQuery or DOM element
// should be returned, which will automatically be placed by Select2 within the
// DOM.
//
// @returns A jQuery or DOM element that contains any elements that must be
//   rendered by Select2.
Adapter.render = function () {
  return $jq;
};

// Bind to any Select2 or DOM events.
//
// @param container The Select2 object that is bound to the jQuery element.  You
//   can listen to Select2 events with `on` and trigger Select2 events using the
//   `trigger` method.
// @param $container The jQuery DOM node that all default adapters will be
//   rendered within.
Adapter.bind = function (container, $container) { };

// Position the DOM element within the Select2 DOM container, or in another
// place. This allows adapters to be located outside of the Select2 DOM,
// such as at the end of the document or in a specific place within the Select2
// DOM node.
//
// Note: This method is not called on data adapters.
//
// @param $rendered The rendered DOM element that was returned from the call to
//   `render`. This may have been modified by Select2, but the root element
//   will always be the same.
// @param $defaultContainer The default container that Select2 will typically
//   place the rendered DOM element within. For most adapters, this is the
//   Select2 DOM element.
Adapter.position = function ($rendered, $defaultContainer) { };

// Destroy any events or DOM elements that have been created.
// This is called when `select2("destroy")` is called on an element.
Adapter.destroy = function () { };
```

### `SelectionAdapter`

The selection is what is shown to the user as a replacement of the standard `<select>` box. It controls the display of the selection option(s), as well anything else that needs to be embedded within the container, such as a search box.

Adapters that will be used to override the default `selectionAdapter` must implement the `update` method as well:

```
// Update the selected data.
//
// @param data An array of data objects that have been generated by the data
//   adapter. If no objects should be selected, an empty array will be passed.
//
// Note: An array will always be passed into this method, even if Select2 is
// attached to a source which only accepts a single selection.
SelectionAdapter.update = function (data) { };
```

### `DataAdapter`

The data set is what Select2 uses to generate the possible results that can be selected, as well as the currently selected results.

Adapters that will be used to override the default `dataAdapter`  must implement the `current` and `query` methods as well:

```
// Get the currently selected options. This is called when trying to get the
// initial selection for Select2, as well as when Select2 needs to determine
// what options within the results are selected.
//
// @param callback A function that should be called when the current selection
//   has been retrieved. The first parameter to the function should be an array
//   of data objects.
DataAdapter.current = function (callback) {
  callback(currentData);
}

// Get a set of options that are filtered based on the parameters that have
// been passed on in.
//
// @param params An object containing any number of parameters that the query
//   could be affected by. Only the core parameters will be documented.
// @param params.term A user-supplied term. This is typically the value of the
//   search box, if one exists, but can also be an empty string or null value.
// @param params.page The specific page that should be loaded. This is typically
//   provided when working with remote data sets, which rely on pagination to
//   determine what objects should be displayed.
// @param callback The function that should be called with the queried results.
DataAdapter.query = function (params, callback) {
  callback(queryiedData);
}
```

## Decorators

Select2 uses [decorators](https://en.wikipedia.org/wiki/Decorator_pattern) to expose the functionality of adapters through its [configuration options](/configuration).

You can apply a decorator to an adapter using the `Utils.Decorate` method provided with Select2:

```
$.fn.select2.amd.require(
    ["select2/utils", "select2/selection/single", "select2/selection/placeholder"],
    function (Utils, SingleSelection, Placeholder) {
  var CustomSelectionAdapter = Utils.Decorate(SingleSelection, Placeholder);
});
```

>>> All core options that use decorators or adapters will clearly state it in the "Decorator" or "Adapter" part of the documentation. Decorators are typically only compatible with a specific type of adapter, so make sure to note what adapter is given.

## AMD Compatibility

You can find more information on how to integrate Select2 with your existing AMD-based project [here](/getting-started/builds-and-modules).  Select2 automatically loads some modules when the adapters are being automatically constructed, so those who are using Select2 with a custom AMD build using their own system may need to specify the paths that are generated to the Select2 modules.
---
title: Selection
taxonomy:
    category: docs
---

Select2 provides the `SingleSelection` and `MultipleSelection` adapters as default implementations of the `SelectionAdapter` for single- and multi-select controls, respectively.  Both `SingleSelection` and `MultipleSelection` extend the base `BaseSelection` adapter.

The selection adapter can be overridden by assigning a custom adapter to the `selectionAdapter` configuration option.

`select2/selection`

## Decorators

### `Placeholder` and `HidePlaceholder`

**AMD Modules:**

`select2/selection/placeholder`
`select2/dropdown/hidePlaceholder`

These decorators implement Select2's [placeholder](/placeholders) features.


### `AllowClear`

**AMD Modules:**

`select2/selection/allowClear`

This decorator implements [clearable selections](/selections#clearable-selections) as exposed through the `allowClear` option.

### `EventRelay`

**AMD Modules:**

`select2/selection/eventRelay`

Select2 has an internal event system that is used to notify parts of the component that state has changed, as well as an adapter that allows some of these events to be relayed to the outside word.
---
title: Array
taxonomy:
    category: docs
---

The `ArrayAdapter` implements support for creating results based on an [array of data objects](/data-sources/arrays).

**AMD Modules:**

`select2/data/array`
---
title: Ajax
taxonomy:
    category: docs
---

The `AjaxAdapter` implements support for creating results [from remote data sources using AJAX requests](/data-sources/ajax).

**AMD Modules:**

`select2/data/ajax`
---
title: SelectAdapter
taxonomy:
    category: docs
---

Select2 provides the `SelectAdapter` as a default implementation of the `DataAdapter` adapter.  It extends `BaseAdapter`.

This adapter can be overridden by assigning a custom adapter to the `dataAdapter` configuration option.
 
**AMD Modules:**

- `select2/data/base`
- `select2/data/select`

## Decorators

### `Tags`

This decorator implements the [tagging](/tagging) feature.

**AMD Modules:**

`select2/data/tags`
  
### `MinimumInputLength`

This decorator implements the [minimum search term length](/searching#minimum-search-term-length) feature as exposed through the `minimumInputLength` configuration option.

**AMD Modules:**

`select2/data/minimumInputLength`

### `MaximumInputLength`

This decorator implements the [maximum search term length](/searching#maximum-search-term-length) feature as exposed through the `maximumInputLength` configuration option.

**AMD Modules:**

`select2/data/maximumInputLength`

### `InitSelection`

This decorator provides backwards compatibility for the `initSelection` callback in version 3.5.

In the past, Select2 required an option called `initSelection` that was defined whenever a custom data source was being used, allowing for the initial selection for the component to be determined. This has been replaced by the `current` method on the data adapter.

**AMD Modules:**

`select2/compat/initSelection"`

### `Query`

This decorator provides backwards compatibility for the `query` callback in version 3.5.

**AMD Modules:**

`select2/compat/query`

### `InputData`

This decorator implements backwards compatibility with version 3.5's `<input type="hidden" >` elements.

In past versions of Select2, a `<select>` element could only be used with a limited subset of options. An `<input type="hidden" >` tag was required instead, which did not allow for a graceful fallback for users who did not have JavaScript enabled. Select2 now supports the `<select>` element for all options, so it is no longer required to use `<input />` elements with Select2.

**AMD Modules:**

`select2/compat/inputData`
---
title: Results
taxonomy:
    category: docs
---

The `ResultsAdapter` controls the list of results that the user can select from.

The results adapter can be overridden by assigning a custom adapter to the `resultsAdapter` configuration option.  While the results adapter does not define any additional methods that must be implemented, it makes extensive use of the Select2 event system for controlling the display of results and messages.
 
**AMD Modules:**

`select2/results`

## Decorators

### `SelectOnClose`

This decorator implements [automatic selection](/dropdown#automatic-selection) of the highlighted option when the dropdown is closed.

**AMD Modules:**

`select2/dropdown/selectOnClose`
---
title: Dropdown
taxonomy:
    category: docs
---

The dropdown adapter defines the main container that the dropdown should be held in.  Select2 allows you to change the way that the dropdown works, allowing you to do anything from attach it to a different location in the document or add a search box.

It is common for decorators to attach to the `render` and `position` methods to alter how the dropdown is altered and positioned.

This adapter can be overridden by assigning a custom adapter to the `dropdownAdapter` configuration option.

`select2/dropdown`

## Decorators

### `AttachBody`

This decorator implements the standard [`dropdownParent`](/dropdown#dropdown-placement) method of attaching the dropdown.

**AMD Modules:**

`select2/dropdown/attachBody`

### `AttachContainer`

When this decorator is loaded, Select2 can place the dropdown directly after the selection container, so it will appear in the same location within the DOM as the rest of Select2.

**AMD Modules:**

`select2/dropdown/attachContainer`

>>>> **Check your build.** This module is only included in the [full builds](/getting-started/builds-and-modules) of Select2.

### `DropdownSearch`

This decorator implements the [search box that is displayed at the top of the dropdown](/searching).

**AMD Modules:**

`select2/dropdown/search`

### `MinimumResultsForSearch`

This decorator implements the [`minimumResultsForSearch` configuration option](/searching#limiting-display-of-the-search-box-to-large-result-sets).

**AMD Modules:**

`select2/dropdown/minimumResultsForSearch`

### `CloseOnSelect`

This decorator implements the [`closeOnSelect` configuration option](/dropdown#forcing-the-dropdown-to-remain-open-after-selection).

`select2/dropdown/closeOnSelect`
---
title: Built-in adapters
taxonomy:
    category: docs
---

This section describes the built-in adapters for Select2, as well as the decorators they use to expose their functionality.
---
title: Advanced
taxonomy:
    category: docs
---

# Advanced Features and Developer Guide
---
title: What's new in 4.0
taxonomy:
    category: docs
---

The 4.0 release of Select2 is the result of three years of working on the code base and watching where it needs to go. At the core, it is a full rewrite that addresses many of the extensibility and usability problems that could not be addressed in previous versions.

This release contains many breaking changes, but easy-upgrade paths have been created as well as helper modules that will allow for backwards compatibility to be maintained with past versions of Select2. Upgrading **will** require you to read the release notes carefully, but the migration path should be relatively straightforward. You can view a list of the most common changes that you will need to make [in the release notes](https://github.com/select2/select2/releases).

The notable features of this new release include:

- A more flexible plugin framework that allows you to override Select2 to behave exactly how you want it to.
- Consistency with standard `<select>` elements for all data adapters, removing the need for hidden `<input>` elements.
- A new build system that uses AMD to keep everything organized.
- Less specific selectors allowing for Select2 to be styled to fit the rest of your application.

## Plugin system

Select2 now provides interfaces that allow for it to be easily extended, allowing for anyone to create a plugin that changes the way Select2 works.  This is the result of Select2 being broken into four distinct sections, each of which can be extended and used together to create your unique Select2.

The adapters implement a consistent interface that is documented in the [advanced chapter](/advanced/adapters-and-decorators), allowing you to customize Select2 to do exactly what you are looking for. Select2 is designed such that you can mix and match plugins, with most of the core options being built as decorators that wrap the standard adapters.

## AMD-based build system

Select2 now uses an [AMD-based build system](https://en.wikipedia.org/wiki/Asynchronous_module_definition), allowing for builds that only require the parts of Select2 that you need.  While a custom build system has not yet been created, Select2 is open source and will gladly accept a pull request for one.

Select2 includes the minimal [almond](https://github.com/jrburke/almond) AMD loader, but a custom `select2.amd.js` build is available if you already use an AMD loader. The code base (available in the `src` directory) also uses AMD, allowing you to include Select2 in your own build system and generate your own builds alongside your existing infrastructure.

The AMD methods used by Select2 are available as `jQuery.fn.select2.amd.define()/require()`, allowing you to use the included almond loader. These methods are primarily used by the translations, but they are the recommended way to access custom modules that Select2 provides.
---
title: Migrating from Select2 3.5
taxonomy:
    category: docs
---

Select2 offers limited backwards compatibility with the previous 3.5.x release line, allowing people to more efficiently transfer across releases and get the latest features. For many of the larger changes, such as the change in how custom data adapters work, compatibility modules were created that will be used to assist in the upgrade process. It is not recommended to rely on these compatibility modules as they will eventually be removed in future releases, but they make upgrading easier for major changes.

If you use the full build of Select2 (`select2.full.js`), you will be automatically notified of the major breaking changes, and [compatibility modules](/upgrading/backwards-compatibility) will be automatically applied to ensure that your code still behaves how you were expecting.

The compatibility modules are only included in the [full builds](/getting-started/builds-and-modules) of Select2. These files end in `.full.js`, and the compatibility modules are prefixed with `select2/compat`.

## No more hidden input tags

In past versions of Select2, an `<input type="hidden">` tag was recommended if you wanted to do anything advanced with Select2, such as work with remote data sources or allow users to add their own tags. This had the unfortunate side-effect of servers not receiving the data from Select2 as an array, like a standard `<select>` element does, but instead sending a string containing the comma-separated strings. The code base ended up being littered with special cases for the hidden input, and libraries using Select2 had to work around the differences it caused.

In Select2 4.0, the `<select>` element supports all core options, and support for the old `<input type="hidden">` has been deprecated. This means that if you previously declared an AJAX field with some pre-selected options that looked like:

```
<input type="hidden" name="select-boxes" value="1,2,4,6" />
```

It will need to be recreated as a `<select>` element with some `<option>` tags that have `value` attributes that match the old value:

```
<select name="select-boxes" multiple="multiple">
    <option value="1" selected="selected">Select2</option>
    <option value="2" selected="selected">Chosen</option>
    <option value="4" selected="selected">selectize.js</option>
    <option value="6" selected="selected">typeahead.js</option>
</select>
```

The options that you create should have `selected="selected"` set so Select2 and the browser knows that they should be selected. The `value` attribute of the option should also be set to the value that will be returned from the server for the result, so Select2 can highlight it as selected in the dropdown. The text within the option should also reflect the value that should be displayed by default for the option.

## Advanced matching of searches

In past versions of Select2 the `matcher` callback processed options at every level, which limited the control that you had when displaying results, especially in cases where there was nested data. The `matcher` function was only given the individual option, even if it was a nested options, without any context.

With the new [matcher function](/searching), only the root-level options are matched and matchers are expected to limit the results of any children options that they contain. This allows developers to customize how options within groups can be displayed, and modify how the results are returned.
 
### Wrapper for old-style `matcher` callbacks

For backwards compatibility, a wrapper function has been created that allows old-style matcher functions to be converted to the new style. 

This wrapper function is only bundled in the [full version of Select2](/getting-started/builds-and-modules).  You can retrieve the function from the `select2/compat/matcher` module, which should just wrap the old matcher function.

<div class="s2-example">
    <select class="js-example-matcher-compat js-states form-control"></select>
</div>

<pre data-fill-from=".js-code-example-matcher-compat"></pre>

<script type="text/javascript" class="js-code-example-matcher-compat">

function matchStart (term, text) {
  if (text.toUpperCase().indexOf(term.toUpperCase()) == 0) {
    return true;
  }

  return false;
}

$.fn.select2.amd.require(['select2/compat/matcher'], function (oldMatcher) {
  $(".js-example-matcher-compat").select2({
    matcher: oldMatcher(matchStart)
  })
});

</script>

>>>> This will work for any matchers that only took in the search term and the text of the option as parameters. If your matcher relied on the third parameter containing the jQuery element representing the original `<option>` tag, then you may need to slightly change your matcher to expect the full JavaScript data object being passed in instead. You can still retrieve the jQuery element from the data object using the `data.element` property.

## More flexible placeholders

In the most recent versions of Select2, placeholders could only be applied to the first (typically the default) option in a `<select>` if it was blank. The `placeholderOption` option was added to Select2 to allow users using the `select` tag to select a different option, typically an automatically generated option with a different value.

The [`placeholder` option](/placeholders) can now take an object as well as just a string. This replaces the need for the old `placeholderOption`, as now the `id` of the object can be set to the `value` attribute of the `<option>` tag.

For a select that looks like the following, where the first option (with a value of `-1`) is the placeholder option:

```
<select>
    <option value="-1" selected="selected">Select an option</option>
    <option value="1">Something else</option>
</select>
```

You would have previously had to get the placeholder option through the `placeholderOption`, but now you can do it through the `placeholder` option by setting an `id`.

```
$("select").select2({
    placeholder: {
        id: "-1",
        placeholder: "Select an option"
    }
});
```

And Select2 will automatically display the placeholder when the value of the select is `-1`, which it will be by default. This does not break the old functionality of Select2 where the placeholder option was blank by default.

## Display reflects the actual order of the values

In past versions of Select2, choices were displayed in the order that they were selected. In cases where Select2 was used on a `<select>` element, the order that the server received the selections did not always match the order that the choices were displayed, resulting in confusion in situations where the order is important.

Select2 will now order selected choices in the same order that will be sent to the server.

## Changed method and option names

When designing the future option set for Select2 4.0, special care was taken to ensure that the most commonly used options were brought over.  For the most part, the commonly used options of Select2 can still be referenced under their previous names, but there were some changes which have been noted.

### Removed the requirement of `initSelection`

>>>> **Deprecated in Select2 4.0.** This has been replaced by another option and is only available in the [full builds](/getting-started/builds-and-modules) of Select2.

In the past, whenever you wanted to use a custom data adapter, such as AJAX or tagging, you needed to help Select2 out in determining the initial
values that were selected. This was typically done through the `initSelection` option, which took the underlying data of the input and converted it into data objects that Select2 could use.

This is now handled by [the data adapter](/advanced/default-adapters/data) in the `current` method, which allows Select2 to convert the currently
selected values into data objects that can be displayed. The default implementation converts the text and value of `option` elements into data objects, and is probably suitable for most cases. An example of the old `initSelection` option is included below, which converts the value of the selected options into a data object with both the `id` and `text` matching the selected value.

```
{
    initSelection : function (element, callback) {
        var data = [];
        $(element.val()).each(function () {
            data.push({id: this, text: this});
        });
        callback(data);
    }
}
```

When using the new `current` method of the custom data adapter, **this method is called any time Select2 needs a list** of the currently selected options. This is different from the old `initSelection` in that it was only called once, so it could suffer from being relatively slow to process the data (such as from a remote data source).

```
$.fn.select2.amd.require([
    'select2/data/array',
    'select2/utils'
], function (ArrayData, Utils) {
    function CustomData ($element, options) {
        CustomData.__super__.constructor.call(this, $element, options);
    }

    Utils.Extend(CustomData, ArrayData);

    CustomData.prototype.current = function (callback) {
        var data = [];
        var currentVal = this.$element.val();

        if (!this.$element.prop('multiple')) {
            currentVal = [currentVal];
        }

        for (var v = 0; v < currentVal.length; v++) {
            data.push({
                id: currentVal[v],
                text: currentVal[v]
            });
        }

        callback(data);
    };

    $("#select").select2({
        dataAdapter: CustomData
    });
});
```

The new `current` method of the data adapter works in a similar way to the old `initSelection` method, with three notable differences. The first, and most important, is that **it is called whenever the current selections are needed** to ensure that Select2 is always displaying the most accurate and up to date data. No matter what type of element Select2 is attached to, whether it supports a single or multiple selections, the data passed to the callback **must be an array, even if it contains one selection**.

The last is that there is only one parameter, the callback to be executed with the latest data, and the current element that Select2 is attached to is available on the class itself as `this.$element`.

If you only need to load in the initial options once, and otherwise will be letting Select2 handle the state of the selections, you don't need to use a custom data adapter. You can just create the `<option>` tags on your own, and Select2 will pick up the changes.

```
var $element = $('select').select2(); // the select element you are working with

var $request = $.ajax({
    url: '/my/remote/source' // wherever your data is actually coming from
});

$request.then(function (data) {
    // This assumes that the data comes back as an array of data objects
    // The idea is that you are using the same callback as the old `initSelection`

    for (var d = 0; d < data.length; d++) {
        var item = data[d];

        // Create the DOM option that is pre-selected by default
        var option = new Option(item.text, item.id, true, true);

        // Append it to the select
        $element.append(option);
    }

    // Update the selected options that are displayed
    $element.trigger('change');
});
```

### Custom data adapters instead of `query`

>>>> **Deprecated in Select2 4.0.** This has been replaced by another option and is only available in the [full builds](/getting-started/builds-and-modules) of Select2.

[In the past](http://select2.github.io/select2/#data), any time you wanted to hook Select2 up to a different data source you would be required to implement custom `query` and `initSelection` methods. This allowed Select2 to determine the initial selection and the list of results to display, and it would handle everything else internally, which was fine more most people.

The custom `query` and `initSelection` methods have been replaced by [custom data adapters](/advanced/default-adapters/data) that handle how Select2 stores and retrieves the data that will be displayed to the user. An example of the old `query` option is provided below, which is
[the same as the old example](http://select2.github.io/select2/#data), and it generates results that contain the search term repeated a certain number of times.

```
{
    query: function (query) {
        var data = {results: []}, i, j, s;
        for (i = 1; i < 5; i++) {
            s = "";
            for (j = 0; j < i; j++) {
                s = s + query.term;
            }
            data.results.push({
                id: query.term + i,
                text: s
            });
        }
        query.callback(data);
    }
}
```
This has been replaced by custom data adapters which define a similarly named `query` method. The comparable data adapter is provided below as an example.

```
$.fn.select2.amd.require([
'select2/data/array',
'select2/utils'
], function (ArrayData, Utils) {
    function CustomData ($element, options) {
        CustomData.__super__.constructor.call(this, $element, options);
    }

    Utils.Extend(CustomData, ArrayData);

    CustomData.prototype.query = function (params, callback) {
        var data = {
            results: []
        };

        for (var i = 1; i < 5; i++) {
            var s = "";

            for (var j = 0; j < i; j++) {
                s = s + params.term;
            }

            data.results.push({
                id: params.term + i,
                text: s
            });
        }

        callback(data);
    };

    $("#select").select2({
        dataAdapter: CustomData
    });
}
```

The new `query` method of the data adapter is very similar to the old `query` option that was passed into Select2 when initializing it. The old `query` argument is mostly the same as the new `params` that are passed in to query on, and the callback that should be used to return the results is now passed in as the second parameter.

### Renamed templating options

Select2 previously provided multiple options for formatting the results list and selected options, commonly referred to as "formatters", using the `formatSelection` and `formatResult` options. As the "formatters" were also used for things such as localization, [which has also changed](#renamed-translation-options), they have been renamed to `templateSelection` and `templateResult` and their signatures have changed as well.

You should refer to the updated documentation on templates for [results](/dropdown) and [selections](/selections) when migrating from previous versions of Select2.

### Renamed `createSearchChoice`

This method has been renamed to `createTag`. You should refer to the documentation on [option creation](/tagging#tag-properties) when migrating from previous versions of Select2.

The `createSearchChoicePosition` option has been removed in favor of the `insertTag` function. New tags are added to the bottom of the list by default.
```
insertTag: function (data, tag) {
  // Insert the tag at the end of the results
  data.push(tag);
}
```

### Renamed `selectOnBlur`

This method has been renamed to `selectOnClose`.

### The `id` and `text` properties are strictly enforced

When working with array and AJAX data in the past, Select2 allowed a custom `id` function or attribute to be set in various places, ranging from the initialization of Select2 to when the remote data was being returned. This allowed Select2 to better integrate with existing data sources that did not necessarily use the `id` attribute to indicate the unique identifier for an object.

Select2 no longer supports a custom `id` or `text` to be used, but provides integration points for converting to the expected format:

#### When working with array data

Select2 previously supported defining array data as an object that matched the signature of an AJAX response. A `text` property could be specified that would map the given property to the `text` property on the individual objects. You can now do this when initializing Select2 by using the following jQuery code to map the old `text` and `id` properties to the new ones.

```
var data = $.map([
    {
        pk: 1,
        word: 'one'
    },
    {
        pk: 2,
        word: 'two'
    }
], function (obj) {
    obj.id = obj.id || obj.pk;
    obj.text = obj.text || obj.word;

    return obj;
});
```

This will result in an array of data objects that have the `id` properties that match the existing `pk` properties and `text` properties that match the existing `word` properties.

#### When working with remote data

The same code that was given above can be used in the `processResults` method of an AJAX call to map properties there as well.

### Renamed translation options

In previous versions of Select2, the default messages provided to users could be localized to fit the language of the website that it was being used on. Select2 only comes with the English language by default, but provides [community-contributed translations](/i18n) for many common languages. Many of the formatters have been moved to the `language` option and the signatures of the formatters have been changed to handle future additions.

### Declaring options using `data-*` attributes

In the past, Select2 has only supported declaring a subset of options using `data-*` attributes. Select2 now supports declaring all options using the attributes, using [the format specified in the documentation](/configuration/data-attributes).

You could previously declare the URL that was used for AJAX requests using the `data-ajax-url` attribute. While Select2 still allows for this, the new attribute that should be used is the `data-ajax--url` attribute. Support for the old attribute will be removed in Select2 4.1.

Although it was not documented, a list of possible tags could also be provided using the `data-select2-tags` attribute and passing in a JSON-formatted array of objects for tags. As the method for specifying tags has changed in 4.0, you should now provide the array of objects using the `data-data` attribute, which maps to [the array data](/data-sources/arrays) option. You should also enable tags by setting `data-tags="true"` on the object, to maintain the ability for users to create their own options as well.

If you previously declared the list of tags as:

```
<select data-select2-tags='[{"id": "1", "text": "One"}, {"id": "2", "text": "Two"}]'></select>
```

...then you should now declare it as...

```
<select data-data='[{"id": "1", "text": "One"}, {"id": "2", "text": "Two"}]' data-tags="true"></select>
```

## Deprecated and removed methods

As Select2 now uses a `<select>` element for all data sources, a few methods that were available by calling `.select2()` are no longer required.

### `.select2("val")`

The `"val"` method has been deprecated and will be removed in Select2 4.1. The deprecated method no longer includes the `triggerChange` parameter.

You should directly call `.val` on the underlying `<select>` element instead. If you needed the second parameter (`triggerChange`), you should also call `.trigger("change")` on the element.

```
$("select").val("1").trigger("change"); // instead of $("select").select2("val", "1");
```

### `.select2("enable")`

Select2 will respect the `disabled` property of the underlying select element. In order to enable or disable Select2, you should call `.prop('disabled', true/false)` on the `<select>` element. Support for the old methods will be completely removed in Select2 4.1.

```
$("select").prop("disabled", true); // instead of $("select").enable(false);
```
---
title: Upgrading
taxonomy:
    category: docs
---

# Upgrading Select2
