module Main (main) where

import Lib
import System.Environment

main :: IO ()
main = do
  (dataDir : _) <- getArgs
  putStrLn $ "Processing files in: " <> dataDir
  processFiles dataDir
