import { describe, it, expect } from "vitest";
import { INITIAL, applyEvent } from "../src/pipeline.js";

describe("applyEvent", () => {
  it("segment marks segment done and detect running", () => {
    const s = applyEvent(INITIAL, { stage: "segment", patches: 5 });
    expect(s.stages.segment).toBe("done");
    expect(s.stages.detect).toBe("running");
    expect(s.patches).toBe(5);
  });

  it("detect stores candidates and starts verify", () => {
    const ev = { stage: "detect", count: 2, candidates: [{ crop_url: "/a" }, { crop_url: "/b" }] };
    const s = applyEvent(INITIAL, ev);
    expect(s.stages.detect).toBe("done");
    expect(s.stages.verify).toBe("running");
    expect(s.candidates).toHaveLength(2);
  });

  it("verify ran=false marks skipped", () => {
    const s = applyEvent(INITIAL, { stage: "verify", ran: false });
    expect(s.stages.verify).toBe("done");
    expect(s.verify.ran).toBe(false);
  });

  it("done sets result", () => {
    const s = applyEvent(INITIAL, {
      stage: "done", found: true, verify_ran: false, bbox: [1, 2, 3, 4], result_url: "/x",
    });
    expect(s.stages.done).toBe("done");
    expect(s.result.found).toBe(true);
    expect(s.result.bbox).toEqual([1, 2, 3, 4]);
  });

  it("error flags running stage as error", () => {
    const mid = applyEvent(INITIAL, { stage: "segment", patches: 1 }); // detect=running
    const s = applyEvent(mid, { stage: "error", message: "boom" });
    expect(s.error).toBe("boom");
    expect(s.stages.detect).toBe("error");
  });
});
