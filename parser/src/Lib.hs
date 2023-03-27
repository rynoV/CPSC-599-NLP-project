{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

module Lib
  ( processFiles,
  )
where

import Control.Applicative
import Control.Monad
import Control.Monad.Extra (mapMaybeM)
import Control.Monad.State
import Data.Aeson qualified as A
import Data.Char (toLower)
import Data.Foldable
import Data.Map qualified as Map
import Data.Text qualified as T
import Data.Text.IO qualified as TIO
import Data.UUID qualified as UUID
import Data.UUID.V4
import GHC.Generics (Generic)
import System.Directory.Extra
import System.Directory.Recursive (getDirFiltered)
import System.FilePath
import Text.Pandoc
import Text.Pandoc.Walk

data LinkData = LinkData {_ldLink :: T.Text, _ldTitle :: T.Text}
  deriving (Show, Eq, Generic)

customOptions :: String -> A.Options
customOptions prefix =
  A.defaultOptions
    { A.fieldLabelModifier = (\(x : xs) -> toLower x : xs) . drop (length prefix)
    }

instance A.ToJSON LinkData where
  toJSON = A.genericToJSON $ customOptions "_ld"
  toEncoding = A.genericToEncoding $ customOptions "_ld"

type LinkStore = Map.Map UUID.UUID LinkData

type LinkHandlerM a = StateT LinkStore IO a

wrapStartText :: T.Text
wrapStartText = "__WRAPSTART"

wrapEndText :: T.Text
wrapEndText = "__WRAPEND__"

processFiles :: FilePath -> IO ()
processFiles dataDir = do
  paths <- getFilePathsInDir dataDir ".html"
  let outDir = dataDir </> "processed"
  for_ paths $ \path -> do
    fileContent <- TIO.readFile path
    when (wrapStartText `T.isInfixOf` fileContent || wrapEndText `T.isInfixOf` fileContent) $
      error $
        "Content of \"" <> path <> "\" contains necessary marker text."
    doc <- parseHtmlFile fileContent
    (doc', linkStore) <- flip runStateT mempty $ walkM handleDocContent doc
    let dir = outDir </> makeRelative dataDir (takeDirectory path)
    createDirectoryIfMissing True dir
    A.encodeFile (dir </> takeBaseName path <> ".linkdata.json") linkStore
    saveAsPlainText doc' (dir </> takeBaseName path)
  putStrLn $ "Processed files to: " <> outDir

contentId :: T.Text
contentId = "sk-page-content-wrapper"

handleDocContent :: Block -> LinkHandlerM Block
handleDocContent (Div attr@(divId, classes, _) blocks)
  | contentId == divId || contentId `elem` classes = Div attr <$> walkM handleLinks blocks
handleDocContent b = pure b

handleLinks :: Inline -> LinkHandlerM Inline
handleLinks (Link _ content (link, title)) = do
  uuid <- liftIO nextRandom
  linkStore <- get
  when (uuid `Map.member` linkStore) $ error "Wow unlucky, duplicate UUIDs generated, try rerunning."
  modify $ Map.insert uuid LinkData {_ldLink = link, _ldTitle = title}
  let uuidText = UUID.toText uuid
  pure $ Span nullAttr (Str (wrapStartText <> uuidText <> "__") : Space : content <> [Space, Str wrapEndText])
handleLinks i = pure i

saveAsPlainText :: Pandoc -> FilePath -> IO ()
saveAsPlainText doc path = do
  w <-
    handleError
      =<< runIO
        ( writePlain
            def
              { writerExtensions = getDefaultExtensions "txt",
                writerWrapText = WrapNone
              }
            doc
        )
  TIO.writeFile (path <> ".txt") w

parseHtmlFile :: T.Text -> IO Pandoc
parseHtmlFile =
  runIO
    . readHtml
      def
        { -- Necessary for links like this to work: <a href="link"><span>Text</span></a>
          readerExtensions = getDefaultExtensions "html"
        }
    >=> handleError

getFilePathsInDir :: FilePath -> String -> IO [FilePath]
getFilePathsInDir dir ext = do
  paths <-
    getDirFiltered
      ( \p ->
          liftA2 (||) (doesDirectoryExist p) $ pure (takeExtension p == ext)
      )
      dir
  flip mapMaybeM paths $ \p -> do
    isDir <- doesDirectoryExist p
    pure $ if isDir then Nothing else Just p
